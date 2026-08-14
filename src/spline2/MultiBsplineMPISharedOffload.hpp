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
 * Only sharing is supported here, not distributing: SplineC2COMPTarget and
 * SplineC2ROMPTarget reach the coefficients through getSplinePtr(), which requires a
 * single block, so the constructor fixes distributed_ranks at 1. Distributing orbitals
 * across ranks additionally needs the SPO evaluation to gather across blocks.
 *
 * Device memory is unchanged: each rank still maps the whole table onto its own device.
 * What shrinks is host memory, by the size of the sharing group, and with it the amount
 * the offload runtime has to register.
 */
template<typename T>
class MultiBsplineMPISharedOffload : public MultiBsplineMPIShared<T>
{
private:
  using Base = MultiBsplineMPIShared<T>;

  /// allocates device space for the shared host coefficients
  std::unique_ptr<MultiBsplineOffloadMapper<T>> mapper_;

public:
  template<typename BCT>
  MultiBsplineMPISharedOffload(const Ugrid grid[3],
                               const BCT& bc,
                               size_t num_splines,
                               std::unique_ptr<Communicate>&& comm_shared)
      : Base(grid, bc, num_splines, std::move(comm_shared), 1)
  {
    // the base constructor has published every block's coefs pointer by now, so the
    // mapper can record them and reserve device space to copy into at finalize time
    mapper_ = std::make_unique<MultiBsplineOffloadMapper<T>>(*this);
    mapper_->mapToDevice();
  }

  /** copy the coefficients to the device and repair the device-side coefs pointer.
   *
   * Called once construction has filled the host coefficients. The single map clause
   * does both jobs, as in MultiBsplineOffload::finalize: coefs is already present on
   * the device from mapToDevice, so mapping it here copies into that allocation, and
   * naming it in the clause is what makes the assignment below store a device address
   * rather than a host one. The fixup is required because the offload kernels in
   * SplineC2COMPTarget dereference spline_m->coefs inside the target region.
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
