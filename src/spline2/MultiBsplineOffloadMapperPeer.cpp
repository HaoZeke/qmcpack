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

bool detail::peerBlockNeedsUpload(const void* device_ptr, size_t bytes)
{
  return device_ptr != nullptr && bytes != 0;
}

template<typename T>
MultiBsplineOffloadMapperPeer<T>::MultiBsplineOffloadMapperPeer(const HostBspline& host_bsplines, Communicate& comm)
    : Base(host_bsplines), comm_(comm)
{
  // IPC coefficient mappings refer to allocations owned by one rank in the group.
  // The peer destructor closes or frees them; the base destructor owns fallback maps.
  Base::owns_coefs_mapping_ = false;
  use_peer_mapping_        = comm.size() > 1 && canShareDeviceMemory(comm);
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

  int current_device      = -1;
  int device_count        = 0;
  bool local_query_failed = cudaGetDevice(&current_device) != cudaSuccess;
  local_query_failed |= cudaGetDeviceCount(&device_count) != cudaSuccess;

  std::array<char, MPI_MAX_PROCESSOR_NAME> local_node{};
  int local_node_length = 0;
  local_query_failed |= MPI_Get_processor_name(local_node.data(), &local_node_length) != MPI_SUCCESS;

  std::array<char, bus_id_size> selected_bus_id_buffer{};
  if (!local_query_failed)
    local_query_failed |= cudaDeviceGetPCIBusId(selected_bus_id_buffer.data(), selected_bus_id_buffer.size(),
                                                current_device) != cudaSuccess;
  const std::string selected_bus_id(selected_bus_id_buffer.data());

  std::vector<std::string> visible_bus_ids;
  visible_bus_ids.reserve(device_count);
  for (int device = 0; device < device_count && !local_query_failed; ++device)
  {
    std::array<char, bus_id_size> visible_bus_id{};
    if (cudaDeviceGetPCIBusId(visible_bus_id.data(), visible_bus_id.size(), device) != cudaSuccess)
      local_query_failed = true;
    else
      visible_bus_ids.emplace_back(visible_bus_id.data());
  }

  if (detail::collectiveFailure(comm, local_query_failed).any_failed)
    return false;

  std::vector<std::string> owner_nodes(comm.size());
  std::vector<std::string> owner_bus_ids(comm.size());
  bool local_broadcast_failed = false;
  for (int owner = 0; owner < comm.size(); ++owner)
  {
    std::array<char, MPI_MAX_PROCESSOR_NAME> owner_node{};
    std::array<char, bus_id_size> owner_bus_id{};
    if (comm.rank() == owner)
    {
      owner_node   = local_node;
      owner_bus_id = selected_bus_id_buffer;
    }

    local_broadcast_failed |=
        MPI_Bcast(owner_node.data(), owner_node.size(), MPI_CHAR, owner, comm.getMPI()) != MPI_SUCCESS;
    local_broadcast_failed |=
        MPI_Bcast(owner_bus_id.data(), owner_bus_id.size(), MPI_CHAR, owner, comm.getMPI()) != MPI_SUCCESS;
    owner_nodes[owner]   = owner_node.data();
    owner_bus_ids[owner] = owner_bus_id.data();
  }

  if (detail::collectiveFailure(comm, local_broadcast_failed).any_failed)
    return false;

  std::vector<int> owner_peer_access(comm.size(), 0);
  const std::string local_node_name(local_node.data(), local_node_length);
  bool local_peer_query_failed = false;
  for (int owner = 0; owner < comm.size(); ++owner)
  {
    if (owner_nodes[owner] != local_node_name)
      continue;

    const auto visible = std::find(visible_bus_ids.begin(), visible_bus_ids.end(), owner_bus_ids[owner]);
    if (visible == visible_bus_ids.end())
      continue;

    const int owner_device = static_cast<int>(std::distance(visible_bus_ids.begin(), visible));
    if (owner_device != current_device)
      local_peer_query_failed |=
          cudaDeviceCanAccessPeer(&owner_peer_access[owner], current_device, owner_device) != cudaSuccess;
    else
      owner_peer_access[owner] = 1;
  }

  if (detail::collectiveFailure(comm, local_peer_query_failed).any_failed)
    return false;

  const int local_access = detail::localPeerTopologyAllowsSharing(local_node_name, selected_bus_id, visible_bus_ids,
                                                                  owner_nodes, owner_bus_ids, owner_peer_access);
  int group_access       = 0;
  if (MPI_Allreduce(&local_access, &group_access, 1, MPI_INT, MPI_MIN, comm.getMPI()) != MPI_SUCCESS)
    throw UniformCommunicateError("MultiBsplineOffloadMapperPeer topology reduction failed!");
  return group_access != 0;
}

template<typename T>
void MultiBsplineOffloadMapperPeer<T>::failCollectivelyIf(bool local_failed, const char* phase, int block)
{
  detail::CollectiveFailure failure{};
  try
  {
    failure = detail::collectiveFailure(comm_, local_failed);
  }
  catch (...)
  {
    // A failed communicator cannot support collective cleanup. Leave the allocations
    // untouched and prevent the destructor from entering another collective.
    peer_mapping_failed_  = true;
    peer_resources_active_ = false;
    throw;
  }

  if (!failure.any_failed)
    return;

  peer_mapping_failed_ = true;
  cleanupPeerMappings();
  throw UniformCommunicateError("MultiBsplineOffloadMapperPeer " + std::string(phase) + " failed for block " +
                                std::to_string(block) + " on rank " + std::to_string(failure.first_failed_rank) +
                                "!");
}

template<typename T>
void MultiBsplineOffloadMapperPeer<T>::cleanupPeerMappings() noexcept
{
  if (!peer_resources_active_)
    return;
  peer_resources_active_ = false;

  const int dev = omp_get_default_device();
  int local_disassociate_failure = 0;
  for (size_t ib = 0; ib < peer_blocks_.size(); ++ib)
  {
    auto& block = peer_blocks_[ib];
    if (!block.associated)
      continue;

    if (omp_target_disassociate_ptr(Base::block_coefs_[ib], dev) == 0)
      block.associated = false;
    else
      local_disassociate_failure = 1;
  }

  int group_disassociate_failure = 1;
  const bool disassociation_complete =
      MPI_Allreduce(&local_disassociate_failure, &group_disassociate_failure, 1, MPI_INT, MPI_MAX, comm_.getMPI()) ==
          MPI_SUCCESS &&
      group_disassociate_failure == 0;

  int local_close_failure = 0;
  if (disassociation_complete)
    for (auto& block : peer_blocks_)
      if (block.imported_handle)
      {
        if (cudaIpcCloseMemHandle(block.device_ptr) == cudaSuccess)
        {
          block.imported_handle = false;
          block.device_ptr      = nullptr;
        }
        else
          local_close_failure = 1;
      }

  int group_close_failure = 1;
  const bool imported_handles_closed =
      disassociation_complete &&
      MPI_Allreduce(&local_close_failure, &group_close_failure, 1, MPI_INT, MPI_MAX, comm_.getMPI()) == MPI_SUCCESS &&
      group_close_failure == 0;

  for (size_t ib = 0; ib < peer_blocks_.size(); ++ib)
    if (peer_blocks_[ib].descriptor_mapped)
    {
      auto* spline_m = &Base::host_bsplines_.getBlock(ib);
      PRAGMA_OFFLOAD("omp target exit data map(delete: spline_m[:1])")
      peer_blocks_[ib].descriptor_mapped = false;
    }

  if (imported_handles_closed)
    for (size_t ib = 0; ib < peer_blocks_.size(); ++ib)
    {
      auto& block = peer_blocks_[ib];
      if (block.owner_allocation && cudaFree(block.device_ptr) == cudaSuccess)
      {
        block.owner_allocation = false;
        block.device_ptr       = nullptr;
      }
    }

  peer_blocks_.clear();
}

template<typename T>
void MultiBsplineOffloadMapperPeer<T>::mapToDevice()
{
  if (!use_peer_mapping_)
  {
    Base::mapToDevice();
    return;
  }

  if (peer_mapping_failed_)
    throw UniformCommunicateError("MultiBsplineOffloadMapperPeer cannot remap after a collective failure!");
  if (peer_resources_active_)
    throw UniformCommunicateError("MultiBsplineOffloadMapperPeer is already mapped!");

  const int dev     = omp_get_default_device();
  const int nranks  = comm_.size();
  const int my_rank = comm_.rank();
  const int nblocks = Base::host_bsplines_.getNumBlocks();
  peer_blocks_.assign(nblocks, PeerBlockState{});
  peer_resources_active_ = true;

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
    auto& block        = peer_blocks_[ib];

    // the descriptor is small and per rank; only the coefficients are worth sharing
    PRAGMA_OFFLOAD("omp target enter data map(to: spline_m[:1])")
    block.descriptor_mapped = true;

    // an empty block has nothing to export: cudaMalloc of zero bytes yields a pointer
    // that cudaIpcGetMemHandle rejects with an invalid argument
    if (bytes == 0)
      continue;

    cudaIpcMemHandle_t handle{};
    bool local_owner_failure = false;
    if (my_rank == owner)
    {
      const auto allocation_status = cudaMalloc(&block.device_ptr, bytes);
      if (allocation_status == cudaSuccess)
      {
        block.owner_allocation = true;
        local_owner_failure = cudaIpcGetMemHandle(&handle, block.device_ptr) != cudaSuccess;
      }
      else
        local_owner_failure = true;
    }
    failCollectivelyIf(local_owner_failure, "allocation/export", ib);

    // raw MPI_Bcast rather than Communicate::bcast, which is instantiated only for
    // the arithmetic types; an IPC handle is an opaque byte blob
    const bool local_broadcast_failure =
        MPI_Bcast(&handle, sizeof(handle), MPI_BYTE, owner, comm_.getMPI()) != MPI_SUCCESS;
    failCollectivelyIf(local_broadcast_failure, "handle broadcast", ib);

    bool local_open_failure = false;
    if (my_rank != owner)
    {
      // cudaIpcMemLazyEnablePeerAccess turns on access to the owner's device on first
      // use. It fails with an invalid argument when the owner's device is not visible
      // to this rank, which is what a one-device-per-task binding produces.
      local_open_failure =
          cudaIpcOpenMemHandle(&block.device_ptr, handle, cudaIpcMemLazyEnablePeerAccess) != cudaSuccess;
      if (!local_open_failure)
        block.imported_handle = true;
    }
    failCollectivelyIf(local_open_failure, "handle import", ib);

    const bool local_association_failure = omp_target_associate_ptr(coefs, block.device_ptr, bytes, 0, dev) != 0;
    if (!local_association_failure)
      block.associated = true;
    failCollectivelyIf(local_association_failure, "pointer association", ib);
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

  if (peer_mapping_failed_ || !peer_resources_active_)
    throw UniformCommunicateError("MultiBsplineOffloadMapperPeer has no usable peer mapping to update!");

  // each block has one physical copy, so its owner writes it and the rest wait rather
  // than pushing identical bytes at memory they do not own
  const int nranks = comm_.size();
  for (int ib = 0; ib < Base::host_bsplines_.getNumBlocks(); ib++)
  {
    bool local_upload_failure = false;
    if (comm_.rank() == ib % nranks)
    {
      auto* spline_m = &Base::host_bsplines_.getBlock(ib);
      auto* coefs    = Base::block_coefs_[ib];
      const size_t bytes = spline_m->coefs_size * sizeof(T);
      if (bytes != 0)
        local_upload_failure =
            !detail::peerBlockNeedsUpload(peer_blocks_[ib].device_ptr, bytes) ||
            cudaMemcpy(peer_blocks_[ib].device_ptr, coefs, bytes, cudaMemcpyHostToDevice) != cudaSuccess;
    }
    failCollectivelyIf(local_upload_failure, "coefficient upload", ib);
  }

  // Each rank owns its descriptor mapping. Store the IPC allocation address in that
  // descriptor so production kernels can dereference spline_m->coefs directly.
  for (int ib = 0; ib < Base::host_bsplines_.getNumBlocks(); ++ib)
  {
    auto* spline_m     = &const_cast<HostBspline&>(Base::host_bsplines_).getBlock(ib);
    auto* device_coefs = static_cast<T*>(peer_blocks_[ib].device_ptr);
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
  cleanupPeerMappings();
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
