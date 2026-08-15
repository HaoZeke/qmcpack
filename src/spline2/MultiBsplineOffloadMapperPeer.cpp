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

#if defined(ENABLE_CUDA)
#include "Platforms/CUDA/CUDAruntime.hpp"
#include <omp.h>
#endif

namespace qmcplusplus
{
template<typename T>
MultiBsplineOffloadMapperPeer<T>::MultiBsplineOffloadMapperPeer(const HostBspline& host_bsplines, Communicate& comm)
    : Base(host_bsplines), comm_(comm)
{
  // the coefficient mappings here come from omp_target_associate_ptr against memory
  // this object may not own, so the base destructor must not try to delete them
  Base::owns_coefs_mapping_ = false;
}

#if defined(ENABLE_CUDA)
template<typename T>
void MultiBsplineOffloadMapperPeer<T>::mapToDevice()
{
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
            std::string("MultiBsplineOffloadMapperPeer: cudaIpcOpenMemHandle failed with '") +
            cudaGetErrorString(err) +
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
}

template<typename T>
MultiBsplineOffloadMapperPeer<T>::~MultiBsplineOffloadMapperPeer()
{
  const int dev = omp_get_default_device();
  for (int ib = 0; ib < Base::host_bsplines_.getNumBlocks(); ib++)
  {
    auto* spline_m = &Base::host_bsplines_.getBlock(ib);
    auto* coefs    = Base::block_coefs_[ib];
    if (device_ptrs_.size() > static_cast<size_t>(ib) && device_ptrs_[ib])
    {
      omp_target_disassociate_ptr(coefs, dev);
      if (comm_.rank() == ib % comm_.size())
        cudaFree(device_ptrs_[ib]);
      else
        cudaIpcCloseMemHandle(device_ptrs_[ib]);
    }
    PRAGMA_OFFLOAD("omp target exit data map(delete: spline_m[:1])")
  }
}

#else // no device runtime that can share memory between processes

template<typename T>
void MultiBsplineOffloadMapperPeer<T>::mapToDevice()
{
  Base::mapToDevice();
}

template<typename T>
void MultiBsplineOffloadMapperPeer<T>::updateToDevice()
{
  Base::updateToDevice();
}

template<typename T>
MultiBsplineOffloadMapperPeer<T>::~MultiBsplineOffloadMapperPeer() = default;

#endif

template class MultiBsplineOffloadMapperPeer<float>;
template class MultiBsplineOffloadMapperPeer<double>;
} // namespace qmcplusplus
