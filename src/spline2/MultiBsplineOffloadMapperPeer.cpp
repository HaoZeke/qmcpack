//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: Rohit Goswami, rgoswami@ieee.org, SURF
//
// File created by: Rohit Goswami, rgoswami@ieee.org, SURF
//////////////////////////////////////////////////////////////////////////////////////

#include "MultiBsplineOffloadMapperPeer.hpp"
#include "Message/UniformCommunicateError.h"
#include "config.h"
#include <algorithm>

#if defined(ENABLE_CUDA)
#include <array>
#include "Platforms/CUDA/CUDAruntime.hpp"
#include "Platforms/Host/OutputManager.h"
#include <omp.h>
#endif

namespace qmcplusplus
{
bool detail::localPeerTopologyAllowsSharing(const std::string& local_node,
                                            const std::string& selected_bus_id,
                                            const std::vector<std::string>& visible_bus_ids,
                                            const std::vector<std::string>& owner_nodes,
                                            const std::vector<std::string>& owner_bus_ids,
                                            const std::vector<int>& owner_peer_access)
{
  if (owner_nodes.size() != owner_bus_ids.size() || owner_nodes.size() != owner_peer_access.size())
    return false;

  for (size_t owner = 0; owner < owner_nodes.size(); ++owner)
  {
    if (owner_nodes[owner] != local_node)
      return false;
    if (std::find(visible_bus_ids.begin(), visible_bus_ids.end(), owner_bus_ids[owner]) == visible_bus_ids.end())
      return false;
    if (owner_bus_ids[owner] != selected_bus_id && owner_peer_access[owner] == 0)
      return false;
  }
  return true;
}

detail::CollectiveFailure detail::collectiveFailure(Communicate& comm, bool local_failed)
{
  const int local_failure_rank = local_failed ? comm.rank() : comm.size();
  int first_failure_rank       = comm.size();
  if (MPI_Allreduce(&local_failure_rank, &first_failure_rank, 1, MPI_INT, MPI_MIN, comm.getMPI()) != MPI_SUCCESS)
    throw UniformCommunicateError("MultiBsplineOffloadMapperPeer failure reduction failed!");
  return {first_failure_rank < comm.size(), first_failure_rank < comm.size() ? first_failure_rank : -1};
}

template<typename T>
MultiBsplineOffloadMapperPeer<T>::MultiBsplineOffloadMapperPeer(const HostBspline& host_bsplines, Communicate& comm)
    : Base(host_bsplines), comm_(comm), use_peer_mapping_(comm.size() > 1 && canShareDeviceMemory(comm))
{
  // IPC coefficient mappings refer to allocations owned by one rank in the group.
  // The peer destructor closes or frees them; the base destructor owns fallback maps.
  Base::owns_coefs_mapping_ = !use_peer_mapping_;
#if defined(ENABLE_CUDA)
  if (comm_.size() > 1 && !use_peer_mapping_ && comm_.rank() == 0)
    app_warning() << "Device coefficient sharing requires one node and mutual peer access to every selected device. "
                     "Using one device mapping per rank.\n";
#endif
}

#if defined(ENABLE_CUDA)
template<typename T>
bool MultiBsplineOffloadMapperPeer<T>::canShareDeviceMemory(Communicate& comm)
{
  constexpr int bus_id_size = 64;

  int current_device;
  int device_count;
  cudaErrorCheck(cudaGetDevice(&current_device), "cudaGetDevice failed in MultiBsplineOffloadMapperPeer!");
  cudaErrorCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed in MultiBsplineOffloadMapperPeer!");

  std::array<char, MPI_MAX_PROCESSOR_NAME> local_node{};
  int local_node_length = 0;
  MPI_Get_processor_name(local_node.data(), &local_node_length);

  std::array<char, bus_id_size> selected_bus_id_buffer{};
  cudaErrorCheck(cudaDeviceGetPCIBusId(selected_bus_id_buffer.data(), selected_bus_id_buffer.size(), current_device),
                 "cudaDeviceGetPCIBusId failed in MultiBsplineOffloadMapperPeer!");
  const std::string selected_bus_id(selected_bus_id_buffer.data());

  std::vector<std::string> visible_bus_ids;
  visible_bus_ids.reserve(device_count);
  for (int device = 0; device < device_count; ++device)
  {
    std::array<char, bus_id_size> visible_bus_id{};
    cudaErrorCheck(cudaDeviceGetPCIBusId(visible_bus_id.data(), visible_bus_id.size(), device),
                   "cudaDeviceGetPCIBusId failed in MultiBsplineOffloadMapperPeer!");
    visible_bus_ids.emplace_back(visible_bus_id.data());
  }

  std::vector<std::string> owner_nodes(comm.size());
  std::vector<std::string> owner_bus_ids(comm.size());
  for (int owner = 0; owner < comm.size(); ++owner)
  {
    std::array<char, MPI_MAX_PROCESSOR_NAME> owner_node{};
    std::array<char, bus_id_size> owner_bus_id{};
    if (comm.rank() == owner)
    {
      owner_node   = local_node;
      owner_bus_id = selected_bus_id_buffer;
    }

    MPI_Bcast(owner_node.data(), owner_node.size(), MPI_CHAR, owner, comm.getMPI());
    MPI_Bcast(owner_bus_id.data(), owner_bus_id.size(), MPI_CHAR, owner, comm.getMPI());
    owner_nodes[owner]   = owner_node.data();
    owner_bus_ids[owner] = owner_bus_id.data();
  }

  std::vector<int> owner_peer_access(comm.size(), 0);
  const std::string local_node_name(local_node.data(), local_node_length);
  for (int owner = 0; owner < comm.size(); ++owner)
  {
    if (owner_nodes[owner] != local_node_name)
      continue;

    const auto visible = std::find(visible_bus_ids.begin(), visible_bus_ids.end(), owner_bus_ids[owner]);
    if (visible == visible_bus_ids.end())
      continue;

    const int owner_device = static_cast<int>(std::distance(visible_bus_ids.begin(), visible));
    if (owner_device != current_device)
      cudaErrorCheck(cudaDeviceCanAccessPeer(&owner_peer_access[owner], current_device, owner_device),
                     "cudaDeviceCanAccessPeer failed in MultiBsplineOffloadMapperPeer!");
    else
      owner_peer_access[owner] = 1;
  }

  const int local_access = detail::localPeerTopologyAllowsSharing(local_node_name, selected_bus_id, visible_bus_ids,
                                                                  owner_nodes, owner_bus_ids, owner_peer_access);
  int group_access       = 0;
  MPI_Allreduce(&local_access, &group_access, 1, MPI_INT, MPI_MIN, comm.getMPI());
  return group_access != 0;
}

template<typename T>
void MultiBsplineOffloadMapperPeer<T>::mapToDevice()
{
  if (!use_peer_mapping_)
  {
    Base::mapToDevice();
    return;
  }

  const int dev     = omp_get_default_device();
  const int nranks  = comm_.size();
  const int my_rank = comm_.rank();
  const int nblocks = Base::host_bsplines_.getNumBlocks();
  device_ptrs_.assign(nblocks, nullptr);

  // Ownership is spread across the ranks rather than parked on rank 0, and that is
  // what makes this worth doing. A single owner holding the whole table leaves its
  // device carrying all of it while the others carry none, and since walkers are
  // balanced across ranks the owner's device stays the binding constraint: the memory
  // freed elsewhere cannot be used. Giving each rank one block puts an equal share on
  // every device, and each reads the rest over the interconnect.
  for (int ib = 0; ib < nblocks; ib++)
  {
    auto* spline_m     = &Base::host_bsplines_.getBlock(ib);
    auto* coefs        = Base::block_coefs_[ib];
    const size_t bytes = spline_m->coefs_size * sizeof(T);
    const int owner    = ib % nranks;

    // the descriptor is small and per rank; only the coefficients are worth sharing
    PRAGMA_OFFLOAD("omp target enter data map(to: spline_m[:1])")

    // an empty block has nothing to export: cudaMalloc of zero bytes yields a pointer
    // that cudaIpcGetMemHandle rejects with an invalid argument
    if (bytes == 0)
      continue;

    void* dptr = nullptr;
    cudaIpcMemHandle_t handle;
    if (my_rank == owner)
    {
      cudaErrorCheck(cudaMalloc(&dptr, bytes), "cudaMalloc failed in MultiBsplineOffloadMapperPeer!");
      cudaErrorCheck(cudaIpcGetMemHandle(&handle, dptr),
                     "cudaIpcGetMemHandle failed in MultiBsplineOffloadMapperPeer!");
    }

    // raw MPI_Bcast rather than Communicate::bcast, which is instantiated only for
    // the arithmetic types; an IPC handle is an opaque byte blob
    MPI_Bcast(&handle, sizeof(handle), MPI_BYTE, owner, comm_.getMPI());

    if (my_rank != owner)
    {
      // cudaIpcMemLazyEnablePeerAccess turns on access to the owner's device on first
      // use. It fails with an invalid argument when the owner's device is not visible
      // to this rank, which is what a one-device-per-task binding produces.
      const auto err = cudaIpcOpenMemHandle(&dptr, handle, cudaIpcMemLazyEnablePeerAccess);
      if (err != cudaSuccess)
        throw UniformCommunicateError(
            std::string("MultiBsplineOffloadMapperPeer: cudaIpcOpenMemHandle failed with '") + cudaGetErrorString(err) +
            "'. Every rank sharing the coefficients must be able to address every device in the "
            "group, so request all the devices on the node rather than binding one per task.");
    }

    if (omp_target_associate_ptr(coefs, dptr, bytes, 0, dev) != 0)
      throw UniformCommunicateError("MultiBsplineOffloadMapperPeer: omp_target_associate_ptr failed!");

    device_ptrs_[ib] = dptr;
  }
}

template<typename T>
void MultiBsplineOffloadMapperPeer<T>::updateToDevice()
{
  if (!use_peer_mapping_)
  {
    Base::updateToDevice();
    return;
  }

  // each block has one physical copy, so its owner writes it and the rest wait rather
  // than pushing identical bytes at memory they do not own
  const int nranks = comm_.size();
  for (int ib = 0; ib < Base::host_bsplines_.getNumBlocks(); ib++)
    if (comm_.rank() == ib % nranks)
    {
      auto* spline_m = &Base::host_bsplines_.getBlock(ib);
      auto* coefs    = Base::block_coefs_[ib];
      cudaErrorCheck(cudaMemcpy(device_ptrs_[ib], coefs, spline_m->coefs_size * sizeof(T), cudaMemcpyHostToDevice),
                     "cudaMemcpy failed in MultiBsplineOffloadMapperPeer!");
    }
  comm_.barrier();

  // Each rank owns its descriptor mapping. Store the IPC allocation address in that
  // descriptor so production kernels can dereference spline_m->coefs directly.
  for (int ib = 0; ib < Base::host_bsplines_.getNumBlocks(); ++ib)
  {
    auto* spline_m     = &const_cast<HostBspline&>(Base::host_bsplines_).getBlock(ib);
    auto* device_coefs = static_cast<T*>(device_ptrs_[ib]);
    if (!device_coefs)
      continue;
    PRAGMA_OFFLOAD("omp target is_device_ptr(device_coefs) map(always, to: spline_m[:1])")
    { spline_m->coefs = device_coefs; }
  }
}

template<typename T>
MultiBsplineOffloadMapperPeer<T>::~MultiBsplineOffloadMapperPeer()
{
  if (!use_peer_mapping_)
    return;

  const int dev = omp_get_default_device();
  for (int ib = 0; ib < Base::host_bsplines_.getNumBlocks(); ib++)
  {
    auto* spline_m = &Base::host_bsplines_.getBlock(ib);
    auto* coefs    = Base::block_coefs_[ib];
    if (device_ptrs_.size() > static_cast<size_t>(ib) && device_ptrs_[ib])
    {
      omp_target_disassociate_ptr(coefs, dev);
      if (comm_.rank() != ib % comm_.size())
        cudaIpcCloseMemHandle(device_ptrs_[ib]);
    }
    PRAGMA_OFFLOAD("omp target exit data map(delete: spline_m[:1])")
  }

  // Importing ranks close every handle before an owner releases its allocation.
  comm_.barrier();
  for (int ib = 0; ib < Base::host_bsplines_.getNumBlocks(); ++ib)
    if (comm_.rank() == ib % comm_.size() && device_ptrs_.size() > static_cast<size_t>(ib) && device_ptrs_[ib])
      cudaFree(device_ptrs_[ib]);
}

#else // no device runtime that can share memory between processes

template<typename T>
bool MultiBsplineOffloadMapperPeer<T>::canShareDeviceMemory(Communicate& comm)
{ return false; }

template<typename T>
void MultiBsplineOffloadMapperPeer<T>::mapToDevice()
{ Base::mapToDevice(); }

template<typename T>
void MultiBsplineOffloadMapperPeer<T>::updateToDevice()
{ Base::updateToDevice(); }

template<typename T>
MultiBsplineOffloadMapperPeer<T>::~MultiBsplineOffloadMapperPeer() = default;

#endif

template class MultiBsplineOffloadMapperPeer<float>;
template class MultiBsplineOffloadMapperPeer<double>;
} // namespace qmcplusplus
