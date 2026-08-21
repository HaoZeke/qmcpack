//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: Rohit Goswami, rgoswami@ieee.org
//
// File created by: Rohit Goswami, rgoswami@ieee.org
//////////////////////////////////////////////////////////////////////////////////////

/** Which loop shape the AB distance table should use, across source counts.
 *
 * mw_evaluate gave one team per target and chunked the sources across that team's
 * threads. A team is far wider than most cells have ions, so the chunked shape leaves
 * nearly every thread idle. Collapsing the target and source loops fills the teams from
 * the product instead.
 *
 * The interesting question is not the small-cell case, where collapsing obviously wins,
 * but the case the chunking was written for: more sources than the 512 chunk width, where
 * the teams were already full and collapsing could only cost. This sweeps the source count
 * across both regimes and reports the ratio, and checks the two shapes agree before timing.
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>

#include "Configuration.h"
#include "OMPTarget/OMPallocator.hpp"
#include "Platforms/PinnedAllocator.h"
#include "OhmmsSoA/VectorSoaContainer.h"
#include "Particle/Lattice/ParticleBConds3DSoa.h"
#include "OMPTarget/OMPTargetMath.hpp"

namespace
{
using namespace qmcplusplus;
using RealType = OHMMS_PRECISION;
template<typename T>
using OffloadAllocator = OMPallocator<T, aligned_allocator<T>>;

constexpr int D          = 3;
constexpr int ChunkSize  = 512; // the width the production chunking used
constexpr int nrepeat    = 40;

/** The body of DTD_BConds<T, 3, SUPERCELL_OPEN>::computeDistancesOffload.
 *
 * Inlined rather than called: it is a non-static member the table reaches through
 * inheritance, and for the open cell it touches no member state, so reproducing it keeps
 * the benchmark free of device-side object construction. The arithmetic is what matters
 * and it is unchanged.
 */
PRAGMA_OFFLOAD("omp declare target")
inline void distance_one(const RealType pos[3],
                         const RealType* restrict R0,
                         int r0_stride,
                         RealType* restrict temp_r,
                         RealType* restrict temp_dr,
                         int padded_size,
                         int iat)
{
  const RealType* restrict px = R0;
  const RealType* restrict py = R0 + r0_stride;
  const RealType* restrict pz = R0 + r0_stride * 2;
  RealType* restrict dx       = temp_dr;
  RealType* restrict dy       = temp_dr + padded_size;
  RealType* restrict dz       = temp_dr + padded_size * 2;

  dx[iat]     = px[iat] - pos[0];
  dy[iat]     = py[iat] - pos[1];
  dz[iat]     = pz[iat] - pos[2];
  temp_r[iat] = std::sqrt(dx[iat] * dx[iat] + dy[iat] * dy[iat] + dz[iat] * dz[iat]);
}
PRAGMA_OFFLOAD("omp end declare target")

/// the shape mw_evaluate used: one team per target, sources chunked across its threads
void chunked(const RealType* src, RealType* out, int num_targets, int num_sources, int num_padded,
             const RealType* tpos)
{
  const int num_teams = (num_sources + ChunkSize - 1) / ChunkSize;
  PRAGMA_OFFLOAD("omp target teams distribute collapse(2) num_teams(num_targets*num_teams)")
  for (int iat = 0; iat < num_targets; ++iat)
    for (int team_id = 0; team_id < num_teams; team_id++)
    {
      auto* r_ptr  = out + iat * num_padded * (D + 1);
      auto* dr_ptr = r_ptr + num_padded;
      const int first = ChunkSize * team_id;
      const int last  = omptarget::min(first + ChunkSize, num_sources);
      RealType pos[D];
      for (int idim = 0; idim < D; idim++)
        pos[idim] = tpos[iat * D + idim];
      PRAGMA_OFFLOAD("omp parallel for")
      for (int iel = first; iel < last; iel++)
        distance_one(pos, src, num_padded, r_ptr, dr_ptr, num_padded, iel);
    }
}

/// the collapsed shape: teams fill from targets times sources
void collapsed(const RealType* src, RealType* out, int num_targets, int num_sources, int num_padded,
               const RealType* tpos)
{
  PRAGMA_OFFLOAD("omp target teams distribute parallel for collapse(2)")
  for (int iat = 0; iat < num_targets; ++iat)
    for (int iel = 0; iel < num_sources; ++iel)
    {
      auto* r_ptr  = out + iat * num_padded * (D + 1);
      auto* dr_ptr = r_ptr + num_padded;
      RealType pos[D];
      for (int idim = 0; idim < D; idim++)
        pos[idim] = tpos[iat * D + idim];
      distance_one(pos, src, num_padded, r_ptr, dr_ptr, num_padded, iel);
    }
}

template<typename F>
double measure(F&& f)
{
  f(); // warm both shapes before either is timed
  const auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < nrepeat; r++)
    f();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

} // namespace

int main(int argc, char** argv)
{
  std::vector<int> source_counts;
  for (int i = 1; i < argc; i++)
    source_counts.push_back(std::atoi(argv[i]));
  if (source_counts.empty())
    source_counts = {2, 8, 33, 128, 512, 1024, 4096, 16384};

  const int num_targets = 512; // stands in for walkers times quadrature knots in one batch

  printf("# AB distance table loop shape, %d targets, %d repeats, chunk width %d\n", num_targets, nrepeat, ChunkSize);
  printf("# %-8s %-12s %-12s %-10s %s\n", "sources", "chunked_ms", "collapsed_ms", "ratio", "agree");

  for (int num_sources : source_counts)
  {
    const int num_padded = getAlignedSize<RealType>(num_sources);
    const size_t out_size = static_cast<size_t>(num_targets) * num_padded * (D + 1);

    Vector<RealType, OffloadAllocator<RealType>> src(static_cast<size_t>(num_padded) * D);
    Vector<RealType, OffloadAllocator<RealType>> tpos(static_cast<size_t>(num_targets) * D);
    Vector<RealType, OffloadAllocator<RealType>> out_a(out_size), out_b(out_size);

    for (int i = 0; i < num_padded * D; i++)
      src[i] = RealType(0.017 * ((i * 37) % 101) - 0.5);
    for (int i = 0; i < num_targets * D; i++)
      tpos[i] = RealType(0.013 * ((i * 53) % 97) - 0.5);
    std::fill(out_a.begin(), out_a.end(), RealType(0));
    std::fill(out_b.begin(), out_b.end(), RealType(0));
    src.updateTo();
    tpos.updateTo();
    out_a.updateTo();
    out_b.updateTo();

    auto* src_ptr  = src.data();
    auto* tpos_ptr = tpos.data();
    auto* a_ptr    = out_a.data();
    auto* b_ptr    = out_b.data();

    chunked(src_ptr, a_ptr, num_targets, num_sources, num_padded, tpos_ptr);
    collapsed(src_ptr, b_ptr, num_targets, num_sources, num_padded, tpos_ptr);
    out_a.updateFrom();
    out_b.updateFrom();

    double worst = 0;
    for (int iat = 0; iat < num_targets; iat++)
      for (int k = 0; k < num_sources; k++)
        for (int f = 0; f < D + 1; f++)
        {
          const size_t idx = static_cast<size_t>(iat) * num_padded * (D + 1) + f * num_padded + k;
          worst            = std::max(worst, std::abs(double(out_a[idx]) - double(out_b[idx])));
        }

    const double ms_a = measure([&] { chunked(src_ptr, a_ptr, num_targets, num_sources, num_padded, tpos_ptr); });
    const double ms_b = measure([&] { collapsed(src_ptr, b_ptr, num_targets, num_sources, num_padded, tpos_ptr); });

    printf("  %-8d %-12.2f %-12.2f %-10.3f %s\n", num_sources, ms_a, ms_b, ms_a / ms_b,
           worst == 0 ? "exact" : "DIFFER");
    if (worst != 0)
      return 2;
  }
  return 0;
}
