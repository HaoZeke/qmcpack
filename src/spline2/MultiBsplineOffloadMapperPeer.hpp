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
// -*- C++ -*-
/**@file MultiBsplineOffloadMapperPeer.hpp
 *
 * define class MultiBsplineOffloadMapperPeer
 */
#ifndef QMCPLUSPLUS_MULTIBSPLINE_OFFLOAD_MAPPER_PEER_HPP
#define QMCPLUSPLUS_MULTIBSPLINE_OFFLOAD_MAPPER_PEER_HPP

#include <vector>
#include "MultiBsplineOffloadMapper.hpp"
#include "Message/Communicate.h"

namespace qmcplusplus
{
/** Map ONE device copy of the coefficients and let every device in the group read it.
 * @tparam T the precision of splines
 *
 * MultiBsplineOffloadMapper allocates device memory per rank, so a node running one
 * rank per device holds one copy of the coefficients per device. For a 586-electron
 * system that is 52.68 GiB on each of four devices, 158 GiB of a table nothing writes
 * after construction, and it is what limits walkers per device rather than any
 * property of the walkers.
 *
 * The coefficients are read-only after construction, so one physical copy can serve
 * the group. The owning rank allocates and exports a handle, the others open it, and
 * every rank binds the resulting pointer to its own host pointer with
 * omp_target_associate_ptr. That is the same call OMPallocator makes under
 * QMC_OFFLOAD_MEM_ASSOCIATED, so the evaluation kernels are untouched: they still
 * dereference spline_m->coefs inside the target region and reach the shared copy.
 *
 * CUDA IPC requires a node-local group in which every rank can see and access each
 * selected device. The mapper checks that invariant collectively and uses ordinary
 * per-rank mappings when it is not satisfied.
 */
template<typename T>
class MultiBsplineOffloadMapperPeer : public MultiBsplineOffloadMapper<T>
{
  using Base        = MultiBsplineOffloadMapper<T>;
  using HostBspline = MultiBsplineBase<T>;

  /// ranks sharing one device copy; must all be on the same node
  Communicate& comm_;
  /// device allocation per block; block ib lives on the device of rank ib % size
  std::vector<void*> device_ptrs_;
  /// whether every rank can open allocations from every owner in this node-local group
  const bool use_peer_mapping_;

public:
  /** Return whether CUDA IPC can connect every rank's selected device.
   *
   * The check covers node locality, device visibility, and peer access. A false
   * result selects the ordinary per-rank mapper.
   */
  static bool canShareDeviceMemory(Communicate& comm);

  MultiBsplineOffloadMapperPeer(const HostBspline& host_bsplines, Communicate& comm);

  ~MultiBsplineOffloadMapperPeer() override;

  /** allocate once for the group and associate it on every rank.
   *
   * Falls back to the base class, one allocation per rank, when the build has no
   * device runtime able to share memory between processes.
   */
  void mapToDevice() override;

  /// only the owner copies; the others are looking at the same memory
  void updateToDevice() override;
};

extern template class MultiBsplineOffloadMapperPeer<float>;
extern template class MultiBsplineOffloadMapperPeer<double>;
} // namespace qmcplusplus

#endif
