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
 * Sharing is what the SPO evaluation can consume today, and it is what the reader asks
 * for. Distributing is accepted by the constructor because the machinery underneath
 * supports it, MultiBsplineMPIShared divides the orbitals into blocks and
 * MultiBsplineOffloadMapper maps and evaluates every block, but it is not yet reachable
 * from a deck: several evaluation paths in SplineC2COMPTarget and SplineC2ROMPTarget
 * still call getSplinePtr(), which throws unless there is exactly one block.
 *
 * With one block, device memory is unchanged: each rank maps the whole table onto its
 * own device, and what shrinks is host memory, by the size of the sharing group. Device
 * memory is the ceiling that actually limits walkers per device, and only distributing
 * moves it.
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
    // The blocks live in one shared window and are adjacent, so naming each block's
    // coefs in its own map clause asks the runtime to extend a mapping it already
    // holds, which it refuses: "explicit extension not allowed", after which any
    // target region dereferencing them gets a null device pointer. The mapper has
    // already allocated the whole span, so push it once and then only repair each
    // descriptor's pointer, with no coefficient mapping in that clause.
    const T* span_begin = nullptr;
    const T* span_end   = nullptr;
    for (size_t ib = 0; ib < Base::getNumBlocks(); ib++)
    {
      const auto& block = Base::getBlock(ib);
      if (block.num_splines == 0)
        continue;
      const T* b = block.coefs;
      const T* e = b + block.coefs_size;
      span_begin = (span_begin == nullptr || b < span_begin) ? b : span_begin;
      span_end   = (span_end == nullptr || e > span_end) ? e : span_end;
    }

    if (span_begin != nullptr)
    {
      auto* coefs             = span_begin;
      const size_t span_count = static_cast<size_t>(span_end - span_begin);
      PRAGMA_OFFLOAD("omp target update to(coefs[:span_count])")
    }

    for (size_t ib = 0; ib < Base::getNumBlocks(); ib++)
    {
      auto* spline_m = &Base::getBlock(ib);
      if (spline_m->num_splines == 0)
        continue;
      auto* coefs = spline_m->coefs;
      // naming coefs in a map clause is what previously made this store a device
      // address, and that is the clause that overlaps. use_device_ptr gets the same
      // address from the span mapping instead, without asking for a second mapping.
      T* dev_coefs = nullptr;
      PRAGMA_OFFLOAD("omp target data use_device_ptr(coefs)")
      { dev_coefs = coefs; }
      PRAGMA_OFFLOAD("omp target map(always, to: spline_m[:1]) firstprivate(dev_coefs)")
      { spline_m->coefs = dev_coefs; }
    }
  }

  ~MultiBsplineMPISharedOffload() override = default;
};

extern template class MultiBsplineMPISharedOffload<float>;
extern template class MultiBsplineMPISharedOffload<double>;
} // namespace qmcplusplus

#endif
