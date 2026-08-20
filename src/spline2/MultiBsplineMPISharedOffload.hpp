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
/**@file MultiBsplineMPISharedOffload.hpp
 *
 * define class MultiBsplineMPISharedOffload
 */
#ifndef QMCPLUSPLUS_MULTIEINSPLINE_MPISHARED_OFFLOAD_HPP
#define QMCPLUSPLUS_MULTIEINSPLINE_MPISHARED_OFFLOAD_HPP

#include <memory>
#include "config.h"
#include "MultiBsplineMPIShared.hpp"
#include "MultiBsplineOffloadMapper.hpp"
#include "MultiBsplineOffloadMapperPeer.hpp"

namespace qmcplusplus
{
/** Spline coefficients held once per group of ranks on the host, mapped to each device.
 * @tparam T the precision of splines
 *
 * MultiBsplineOffload gives every rank its own host allocation and its own device
 * allocation. With one rank per device that is one host copy per device, so a node
 * running four ranks holds four copies of a table none of them writes after
 * construction.
 *
 * MultiBsplineMPIShared already places the coefficients in an MPI-3 shared window, one
 * copy per group of ranks, but has no route to a device. MultiBsplineOffloadMapper
 * already maps an arbitrary host spline onto devices. This joins the two.
 *
 * MultiBsplineMPIShared divides the orbitals into blocks, and
 * MultiBsplineOffloadMapper maps and evaluates each block. Complex offload SPO paths
 * consume distributed blocks directly. Real offload SPO paths require one block.
 *
 * A peer-capable node places each block on one selected device and lets the other ranks
 * read it through CUDA IPC. Ownership rotates across ranks, so matching the number of
 * blocks to the selected devices balances coefficient storage. Configurations without
 * mutual peer access use one complete device mapping per rank.
 */
template<typename T>
class MultiBsplineMPISharedOffload : public MultiBsplineMPIShared<T>
{
private:
  using Base = MultiBsplineMPIShared<T>;

  /// allocates device space for the shared host coefficients
  std::unique_ptr<MultiBsplineOffloadMapper<T>> mapper_;

public:
  /** @param distributed_ranks how many blocks to divide the orbitals into.
   *
   * 1 keeps every orbital in one block. Larger values divide the orbitals into that
   * many blocks so ownership can be balanced across devices.
   */
  template<typename BCT>
  MultiBsplineMPISharedOffload(const Ugrid grid[3],
                               const BCT& bc,
                               size_t num_splines,
                               std::unique_ptr<Communicate>&& comm_shared,
                               unsigned distributed_ranks = 1)
      : Base(grid, bc, num_splines, std::move(comm_shared), distributed_ranks)
  {
    // The base constructor publishes every block's coefficient pointer. Multi-rank
    // groups use a capability-aware peer mapper; single-rank groups use ordinary maps.
    if (Base::getSharingComm().size() > 1)
      mapper_ = std::make_unique<MultiBsplineOffloadMapperPeer<T>>(*this, Base::getSharingComm());
    else
      mapper_ = std::make_unique<MultiBsplineOffloadMapper<T>>(*this);
    mapper_->mapToDevice();
  }

  /** Publish shared coefficients, upload them, and attach descriptor pointers. */
  void finalize() override
  {
    Base::finalize();
    mapper_->updateToDevice();
  }

  /** Evaluate mapped spline values through the mapper owned by this object.
   *
   * This keeps construction, finalization, and evaluation on the same mapping.
   */
  void mw_evaluate_v(int num_pos, T* pos_arr, T* spline_v, size_t walker_stride)
  { mapper_->mw_evaluate_v(num_pos, pos_arr, spline_v, walker_stride); }

  ~MultiBsplineMPISharedOffload() override = default;
};

extern template class MultiBsplineMPISharedOffload<float>;
extern template class MultiBsplineMPISharedOffload<double>;
} // namespace qmcplusplus

#endif
