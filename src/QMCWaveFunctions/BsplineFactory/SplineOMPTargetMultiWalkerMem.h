//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_OFFLOADSHAREDMEM_H
#define QMCPLUSPLUS_OFFLOADSHAREDMEM_H

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>
#include "config.h"
#include "OMPTarget/OffloadAlignedAllocators.hpp"
#include "ResourceCollection.h"

namespace qmcplusplus
{

/** one team's share of one block of a distributed spline table
 *
 * The blocks differ only in how many orbitals they hold and where those orbitals start,
 * and they share one grid, so flattening every (block, team) pair into a single list
 * lets one kernel cover them all: a team reads its record and needs to know nothing
 * about which block the record came from. Without this the evaluation launches a kernel
 * per block, and the launches are what distributing costs.
 */
template<typename ST>
struct SplineBlockTeam
{
  /// device address of the block's coefficients
  const ST* coefs;
  std::intptr_t x_stride;
  std::intptr_t y_stride;
  std::intptr_t z_stride;
  /// the team's range of orbital indices within the block
  int first;
  int last;
  /// where the block's orbitals start in the output
  std::size_t out_offset;
};

/** flatten a spline table's blocks into one team list on the device
 *
 * Call once, after finalize has left each block's device descriptor holding a device
 * address for its coefficients, and outside any threaded region. The list is shared
 * rather than copied so a cloned SPO reads the buffer that was pushed.
 */
template<typename ST, typename SPLINE, typename TEAMPTR>
inline void buildSplineBlockTeams(SPLINE& spline_inst, TEAMPTR& teams)
{
  const size_t num_blocks   = spline_inst.getNumBlocks();
  const auto& block_offsets = spline_inst.getBlockOffsets();
  // the orbitals a team interpolates, matching the chunking the evaluation uses
  constexpr size_t chunk = 512;

  std::vector<SplineBlockTeam<ST>> host_teams;
  for (size_t ib = 0; ib < num_blocks; ib++)
  {
    auto& block = spline_inst.getBlock(ib);
    if (block.num_splines == 0)
      continue;
    // the host descriptor keeps the host address; the device one was repaired to the
    // device address, and use_device_ptr reads that same address without remapping
    auto* coefs   = block.coefs;
    ST* dev_coefs = nullptr;
    PRAGMA_OFFLOAD("omp target data use_device_ptr(coefs)")
    { dev_coefs = coefs; }

    const size_t block_splines = static_cast<size_t>(block.num_splines);
    for (size_t first = 0; first < block_splines; first += chunk)
      host_teams.push_back(SplineBlockTeam<ST>{dev_coefs, block.x_stride, block.y_stride, block.z_stride,
                                               static_cast<int>(first),
                                               static_cast<int>(std::min(first + chunk, block_splines)),
                                               block_offsets[ib]});
  }

  using VecType = typename TEAMPTR::element_type;
  teams         = std::make_shared<VecType>(host_teams.size());
  std::copy(host_teams.begin(), host_teams.end(), teams->begin());
  teams->updateTo();
}



template<typename ST, typename TT>
struct SplineOMPTargetMultiWalkerMem : public Resource
{
  ///team private ratios for reduction, numVP x numTeams
  Matrix<TT, OffloadPinnedAllocator<TT>> mw_ratios_private;
  ///team private ratios and grads for reduction, numVP x numTeams
  Matrix<TT, OffloadPinnedAllocator<TT>> rg_private;
  /// ratio and gradients reduced on the device, filled only by the device entry point so
  /// callers that do not ask for them pay nothing: [nw] and [nw][3] flat
  Vector<TT, OffloadPinnedAllocator<TT>> ratios_device;
  Vector<TT, OffloadPinnedAllocator<TT>> grads_device;
  ///offload scratch space, dynamically resized to the maximal need
  Vector<ST, OffloadPinnedAllocator<ST>> mw_offload_scratch;
  ///result scratch space, dynamically resized to the maximal need
  Vector<TT, OffloadPinnedAllocator<TT>> mw_results_scratch;
  ///position scratch space, used to avoid allocation on the fly and faster transfer
  Vector<ST, OffloadPinnedAllocator<ST>> mw_pos_copy;
  ///multi purpose H2D buffer for mw_evaluateVGLandDetRatioGrads
  Matrix<char, OffloadPinnedAllocator<char>> buffer_H2D;
  ///multi purpose H2D buffer for mw_evaluateDetRatios
  Vector<char, OffloadPinnedAllocator<char>> det_ratios_buffer_H2D;

  SplineOMPTargetMultiWalkerMem() : Resource("SplineOMPTargetMultiWalkerMem") {}

  SplineOMPTargetMultiWalkerMem(const SplineOMPTargetMultiWalkerMem&) : SplineOMPTargetMultiWalkerMem() {}

  std::unique_ptr<Resource> makeClone() const override
  { return std::make_unique<SplineOMPTargetMultiWalkerMem>(*this); }
};
} // namespace qmcplusplus
#endif
