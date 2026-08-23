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
#include "Host/OutputManager.h"
#include <iostream>
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
#if defined(ENABLE_CUDA)
  // the coefficient mappings here come from omp_target_associate_ptr against memory
  // this object may not own, so the base destructor must not try to delete them.
  // Without a device runtime that can share memory this class falls back to the base
  // implementation, which DOES create the mappings, and clearing the flag there would
  // leave them behind: the next object to allocate nearby is then refused with
  // "explicit extension not allowed" under OMP_TARGET_OFFLOAD=mandatory.
  Base::owns_coefs_mapping_ = false;
#endif
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

    // A group of one rank owns every block, so there is nothing to share and the
    // ordinary mapping path is both sufficient and better behaved: associating memory
    // this process already owns leaves the runtime holding the host range after
    // teardown, and a later object allocating nearby is then refused with "explicit
    // extension not allowed".
    if (nranks == 1)
    {
      const T* coefs_local = coefs;
      PRAGMA_OFFLOAD("omp target enter data map(alloc: coefs_local[:spline_m->coefs_size])")
      device_ptrs_[ib] = nullptr; // nothing IPC-owned to release
      continue;
    }

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

  /* What a run gets out of this is invisible otherwise. A node holding one copy of a
   * read only table per device spends that memory on nothing, and it is device memory
   * that limits walkers per device, so say how many copies there are and what the
   * arrangement costs against one copy per rank.
   */
  size_t shared_bytes = 0;
  for (int ib = 0; ib < nblocks; ib++)
    shared_bytes += Base::host_bsplines_.getBlock(ib).coefs_size * sizeof(T);
  const double mib = static_cast<double>(shared_bytes) / (1 << 20);
  if (comm_.rank() == 0)
  {
    if (nranks > 1)
      app_log() << "  Spline coefficients: one device copy of " << mib << " MiB shared across " << nranks
                << " ranks, against " << mib * nranks << " MiB for a copy each." << std::endl;
    else
      app_log() << "  Spline coefficients: " << mib << " MiB on this rank's device, not shared." << std::endl;
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
    if (comm_.size() == 1)
    {
      const T* coefs_local = coefs;
      PRAGMA_OFFLOAD("omp target exit data map(delete: coefs_local[:spline_m->coefs_size])")
    }
    else if (device_ptrs_.size() > static_cast<size_t>(ib) && device_ptrs_[ib])
    {
      // a failed disassociate leaves the runtime holding this host range, and the
      // next object to map an allocation at the same address is refused with
      // "explicit extension not allowed"
      if (const int st = omp_target_disassociate_ptr(coefs, dev); st != 0)
        std::cerr << "MultiBsplineOffloadMapperPeer: omp_target_disassociate_ptr returned " << st
                  << " for block " << ib << std::endl;
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
  /* Nothing can be shared without a device runtime that hands an allocation to another
   * process, so every rank maps its own copy, which is what the ordinary mapper does.
   *
   * Say so. The request came from the input, the host side of it did take effect, and the
   * line that reports it says the coefficients are shared. A caller sizing a run on device
   * memory would read that and be wrong by a factor of the group size, which on a large
   * table is the difference between fitting and not.
   */
  if (comm_.size() > 1 && comm_.rank() == 0)
    app_warning() << "Spline coefficients are shared on the host across " << comm_.size()
                  << " ranks, but this build cannot share a device allocation between processes, so "
                     "each rank still holds its own device copy. Configure with ENABLE_CUDA to share "
                     "the device copy as well." << std::endl;
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
