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
// the allreduce specialisations: Communicate.h declares the template, and the
// int form that reduces the failure flag is defined here
#include "Message/CommOperators.h"
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
    : Base(host_bsplines, false), comm_(comm)
{
  // the coefficient mappings here come from omp_target_associate_ptr against memory
  // this object may not own, so the base destructor must not try to delete them
  Base::owns_coefs_mapping_ = false;
  // the base was told not to map: one allocation per rank is the thing this class
  // exists to avoid, and mapping it first would then have to be undone
  mapToDevice();
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
  // filled by whichever step fails, and reduced across the group below
  std::string local_error;

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
    descriptors_mapped_ = ib + 1;

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
      {
        local_error = std::string("cudaIpcOpenMemHandle failed with '") + cudaGetErrorString(err) +
            "'. Every rank sharing the coefficients must be able to address every device in the "
            "group, so request all the devices on the node rather than binding one per task.";
        break;
      }
    }

    if (omp_target_associate_ptr(coefs, dptr, bytes, 0, dev) != 0)
    {
      local_error = "omp_target_associate_ptr failed";
      break;
    }

    device_ptrs_[ib] = dptr;
  }

  /* Every rank has to agree before any of them throws. A rank that fails to import a
   * handle and throws alone leaves its peers at the next collective, which is a hang
   * rather than an error, and the message that would explain it is on the rank that
   * left. So the failure is reduced first and every rank raises it.
   */
  int failed = local_error.empty() ? 0 : 1;
  comm_.allreduce(failed);
  if (failed)
  {
    // a rank that succeeded has to undo what it did before it throws, or the
    // destructor will not run and the imported handles stay open
    releaseDeviceMappings();
    throw UniformCommunicateError("MultiBsplineOffloadMapperPeer: " +
                                  (local_error.empty()
                                       ? std::string("another rank in the group failed to share the "
                                                     "coefficients on the device; its own message says why")
                                       : local_error));
  }
}

template<typename T>
void MultiBsplineOffloadMapperPeer<T>::releaseDeviceMappings()
{
  /* Teardown order matters and it is collective. An importer's handle refers to the
   * owner's allocation, so every importer has to close before the owner frees;
   * without the barrier a rank can free block ib while a peer still holds it open.
   * Disassociating is rank-local and comes first because it names the host pointer.
   */
  if (released_)
    return;
  released_ = true;

  const int dev     = omp_get_default_device();
  const int nblocks = Base::host_bsplines_.getNumBlocks();

  for (int ib = 0; ib < nblocks; ib++)
    if (ib < static_cast<int>(device_ptrs_.size()) && device_ptrs_[ib])
      omp_target_disassociate_ptr(Base::block_coefs_[ib], dev);

  comm_.barrier();

  for (int ib = 0; ib < nblocks; ib++)
    if (ib < static_cast<int>(device_ptrs_.size()) && device_ptrs_[ib] && comm_.rank() != ib % comm_.size())
      cudaIpcCloseMemHandle(device_ptrs_[ib]);

  comm_.barrier();

  for (int ib = 0; ib < nblocks; ib++)
    if (ib < static_cast<int>(device_ptrs_.size()) && device_ptrs_[ib] && comm_.rank() == ib % comm_.size())
      cudaFree(device_ptrs_[ib]);

  device_ptrs_.assign(nblocks, nullptr);

  // only the descriptors that were mapped: a failure part way through the block loop
  // leaves the rest of them never entered, and deleting one of those is not a no-op
  // on every runtime
  for (int ib = 0; ib < descriptors_mapped_; ib++)
  {
    auto* spline_m = &Base::host_bsplines_.getBlock(ib);
    PRAGMA_OFFLOAD("omp target exit data map(delete: spline_m[:1])")
  }
  descriptors_mapped_ = 0;
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
  releaseDeviceMappings();
}

#else // no device runtime that can share memory between processes

template<typename T>
void MultiBsplineOffloadMapperPeer<T>::mapToDevice()
{
  // the base allocated these, so the base destructor is the one that must delete them
  Base::owns_coefs_mapping_ = true;
  Base::mapToDevice();
}

template<typename T>
void MultiBsplineOffloadMapperPeer<T>::updateToDevice()
{
  Base::updateToDevice();
}

template<typename T>
void MultiBsplineOffloadMapperPeer<T>::releaseDeviceMappings()
{
  // nothing of this class's own to release: the base owns the per-rank mappings
  released_ = true;
}

template<typename T>
MultiBsplineOffloadMapperPeer<T>::~MultiBsplineOffloadMapperPeer() = default;

#endif

template class MultiBsplineOffloadMapperPeer<float>;
template class MultiBsplineOffloadMapperPeer<double>;
} // namespace qmcplusplus
