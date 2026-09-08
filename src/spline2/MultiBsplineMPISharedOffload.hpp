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
 * Both are reachable from a deck on a complex offload build. MultiBsplineMPIShared
 * divides the orbitals into blocks, MultiBsplineOffloadMapper maps and evaluates every
 * block, and every evaluation path in SplineC2C now indexes by block, so
 * SplineSetReader no longer overrides distributed_ranks to 1 there. A real-valued
 * build still does, because SplineC2R reaches the coefficients through getSplinePtr(),
 * which throws unless there is exactly one block.
 *
 * The distinction is worth keeping straight, because the two halves buy different
 * things. With one block per rank, device memory is unchanged: each rank maps the whole
 * table onto its own device and what shrinks is host memory, by the size of the sharing
 * group. Device memory is the ceiling that limits walkers per device, and only
 * distributing moves it.
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
   * 1 shares one copy of every orbital across the group, which is all the SPO
   * evaluation can consume today. Larger values divide the orbitals into that many
   * blocks, which is what would let a multi-device node stop holding the whole table
   * on every device, and the mapper and the base class already handle it. The reader
   * still refuses to ask for more than 1 until every evaluation path stops calling
   * getSplinePtr(), so this parameter exists to be tested rather than deployed.
   */
  template<typename BCT>
  MultiBsplineMPISharedOffload(const Ugrid grid[3],
                               const BCT& bc,
                               size_t num_splines,
                               std::unique_ptr<Communicate>&& comm_shared,
                               unsigned distributed_ranks = 1)
      : Base(grid, bc, num_splines, std::move(comm_shared), distributed_ranks)
  {
    // the base constructor has published every block's coefs pointer by now, so the
    // mapper can record them and reserve device space to copy into at finalize time
    // A group of more than one rank shares the coefficients on the host already, and
    // on a node whose devices can address each other they can share the device copy
    // too. That is the half that matters: the host copy is not what limits walkers per
    // device, the device copy is. One rank per device otherwise, as before.
    if (Base::getSharingComm().size() > 1)
      mapper_ = std::make_unique<MultiBsplineOffloadMapperPeer<T>>(*this, Base::getSharingComm());
    else
      mapper_ = std::make_unique<MultiBsplineOffloadMapper<T>>(*this);
    // Both of those map in their own constructor, so there is nothing to call
    // here. mapToDevice is protected on the base and reaching it through a base
    // pointer would not compile even if there were.
  }

  /** copy the coefficients to the device and repair the device-side coefs pointer.
   *
   * Called once construction has filled the host coefficients. The single map clause
   * does both jobs, as in MultiBsplineOffload::finalize: coefs is already present on
   * the device from mapToDevice, so mapping it here copies into that allocation, and
   * naming it in the clause is what makes the assignment below store a device address
   * rather than a host one. The fixup is required because the offload kernels in
   * SplineC2C dereference spline_m->coefs inside the target region.
   */
  void finalize() override
  {
    for (size_t ib = 0; ib < Base::getNumBlocks(); ib++)
    {
      auto* spline_m = &Base::getBlock(ib);
      auto* coefs    = spline_m->coefs;
      PRAGMA_OFFLOAD("omp target map(always, to: spline_m[:1], coefs[:spline_m->coefs_size])")
      { spline_m->coefs = coefs; }
    }
  }

  ~MultiBsplineMPISharedOffload() override = default;
};

extern template class MultiBsplineMPISharedOffload<float>;
extern template class MultiBsplineMPISharedOffload<double>;
} // namespace qmcplusplus

#endif
