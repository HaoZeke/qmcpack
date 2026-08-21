//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** Time the offloaded B-spline VGL evaluation against the orbital count.
 *
 *  The physics decks small enough to keep beside a developer checkout carry a handful of
 *  orbitals, and at that size the evaluation kernel is pure overhead: the coefficient table
 *  fits in cache and each thread does almost no arithmetic. Production systems carry
 *  hundreds, where the table no longer fits and the access pattern is what costs. A change
 *  to the kernel therefore cannot be judged on those decks, and this fills the gap without
 *  needing a production wavefunction: build a spline of a chosen orbital count, evaluate it
 *  for many positions on the device, and report the time per orbital evaluation.
 *
 *  Run with the orbital counts to sweep, e.g.
 *    benchmark_vgl_offload 8 32 128 512
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <omp.h>

#include "Configuration.h"
#include "spline2/MultiBspline.hpp"
#include "spline2/MultiBsplineOffloadMapper.hpp"
#include "spline2/MultiBsplineVGLH_OMPoffload.hpp"
#include "einspline/bspline_create.h"
#include "OhmmsPETE/OhmmsVector.h"
#include "OMPTarget/OffloadAlignedAllocators.hpp"

using namespace qmcplusplus;

namespace
{
using RealType = OHMMS_PRECISION;

/// grid, boundary conditions and sample data for a periodic spline
struct SplineSetup
{
  Ugrid grid[3];
  BCtype_d bc[3];
  std::vector<double> data;

  SplineSetup(int grid_size)
  {
    for (int d = 0; d < 3; d++)
    {
      grid[d].start = 0.0;
      grid[d].end   = 1.0;
      grid[d].num   = grid_size;
      bc[d].lCode   = PERIODIC;
      bc[d].rCode   = PERIODIC;
      bc[d].lVal    = 0.0;
      bc[d].rVal    = 0.0;
    }
    const double delta = 1.0 / grid_size;
    const double tpi   = 2 * M_PI;
    data.resize(static_cast<size_t>(grid_size) * grid_size * grid_size);
    for (int i = 0; i < grid_size; i++)
      for (int j = 0; j < grid_size; j++)
        for (int k = 0; k < grid_size; k++)
          data[(static_cast<size_t>(i) * grid_size + j) * grid_size + k] =
              std::sin(tpi * delta * i) + std::sin(3 * tpi * delta * j) + std::sin(4 * tpi * delta * k);
  }
};

} // namespace

int main(int argc, char** argv)
{
  constexpr int grid_size = 20;
  constexpr int npos      = 512;  // stands in for walkers times electrons in one batch
  constexpr int nrepeat   = 40;

  std::vector<int> sizes;
  for (int i = 1; i < argc; i++)
    sizes.push_back(std::atoi(argv[i]));
  if (sizes.empty())
    sizes = {8, 32, 128, 512};

  printf("# offloaded B-spline VGL, grid %d^3, %d positions, %d repeats\n", grid_size, npos, nrepeat);
  printf("# %-8s %-12s %-12s %-14s %s\n", "orbitals", "old_ms", "new_ms", "us_per_pos", "ns_per_orbital_eval");

  for (int num_splines : sizes)
  {
    SplineSetup setup(grid_size);
    const size_t padded = getAlignedSize<RealType>(num_splines);

    MultiBspline<RealType> mb(setup.grid, setup.bc, padded);
    UBspline_3d_d* aspline = create_UBspline_3d_d(setup.grid[0], setup.grid[1], setup.grid[2], setup.bc[0],
                                                 setup.bc[1], setup.bc[2], setup.data.data());
    for (int i = 0; i < num_splines; i++)
      mb.set_spline(*aspline, i);
    destroy_Bspline(aspline);

    // Map the coefficients with the production mapper. Without this the coefficient table
    // stays host memory and the target region dereferences an unmapped pointer, which does
    // not fail loudly but does put every coefficient load across the bus. Timing that
    // exaggerates the value of removing loads, so the mapping is what makes the numbers mean
    // anything.
    MultiBsplineOffloadMapper<RealType> mapper(mb);
    mapper.mapToDevice();
    mapper.updateToDevice();

    auto* spline_ptr          = mb.getSplinePtr();
    const size_t scratch_size = padded * 11; // VAL, 3 grads, 6 hessian, LAPL

    Vector<RealType, OffloadPinnedAllocator<RealType>> scratch(scratch_size * npos);
    scratch.updateTo();
    auto* scratch_ptr = scratch.data();

    // a non-diagonal metric, so a wrong contraction cannot hide behind symmetry
    const RealType symGGt[6] = {1.07, 0.13, -0.21, 0.94, 0.08, 1.11};

    // confirm the kernel really runs on the device rather than falling back
    int on_device = -1;
    PRAGMA_OFFLOAD("omp target map(from: on_device)")
    on_device = omp_is_initial_device() ? 0 : 1;
    if (on_device != 1)
    {
      printf("  ERROR: kernel did not run on the device (omp_is_initial_device reported %d)\n", on_device);
      return 2;
    }

    // both paths, same inputs, compared before anything is timed
    {
      Vector<RealType, OffloadPinnedAllocator<RealType>> ref(scratch_size), got(scratch_size);
      ref.updateTo();
      got.updateTo();
      auto* ref_ptr = ref.data();
      auto* got_ptr = got.data();
      PRAGMA_OFFLOAD("omp target teams distribute map(always, from: ref_ptr[0:scratch_size], got_ptr[0:scratch_size])")
      for (int one = 0; one < 1; one++)
      {
        int ix, iy, iz;
        RealType a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
        spline2::computeLocationAndFractional(spline_ptr, RealType(0.31), RealType(0.57), RealType(0.13), ix, iy, iz, a,
                                              b, c, da, db, dc, d2a, d2b, d2c);
        PRAGMA_OFFLOAD("omp parallel for")
        for (size_t sp = 0; sp < padded; sp++)
        {
          spline2offload::evaluate_vgh_impl_v2(spline_ptr, spline_ptr->coefs, ix, iy, iz, static_cast<int>(sp), a, b, c,
                                               da, db, dc, d2a, d2b, d2c, ref_ptr + sp, padded);
          ref_ptr[padded * 10 + sp] = ref_ptr[padded * 4 + sp] * symGGt[0] + ref_ptr[padded * 5 + sp] * symGGt[1] +
              ref_ptr[padded * 6 + sp] * symGGt[2] + ref_ptr[padded * 7 + sp] * symGGt[3] +
              ref_ptr[padded * 8 + sp] * symGGt[4] + ref_ptr[padded * 9 + sp] * symGGt[5];
          spline2offload::evaluate_vgl_impl_v2(spline_ptr, spline_ptr->coefs, ix, iy, iz, static_cast<int>(sp), a, b, c,
                                               da, db, dc, d2a, d2b, d2c, symGGt, got_ptr + sp, padded, padded * 10);
        }
      }
      const int fields[5] = {0, 1, 2, 3, 10}; // value, three gradients, laplacian
      double worst        = 0;
      for (int f : fields)
        for (int sp = 0; sp < num_splines; sp++) // unpadded orbitals only
        {
          const double r = ref[padded * f + sp], g = got[padded * f + sp];
          const double d = std::abs(r - g) / std::max(1.0, std::abs(r));
          if (d > worst)
            worst = d;
        }
      const double tol = sizeof(RealType) == 4 ? 1e-5 : 1e-12;
      printf("  # orbitals=%d worst relative difference old vs new = %.3e (tol %.0e) %s\n", num_splines, worst, tol,
             worst <= tol ? "OK" : "MISMATCH");
      if (worst > tol)
        return 3;
    }

    // the old path: write ten fields, then read the six hessian components back to contract
    auto t0old = std::chrono::steady_clock::now();
    for (int rep = 0; rep < nrepeat; rep++)
    {
      const RealType frac = static_cast<RealType>(rep) / nrepeat;
      PRAGMA_OFFLOAD("omp target teams distribute")
      for (int ip = 0; ip < npos; ip++)
      {
        int ix, iy, iz;
        RealType a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
        const RealType x = frac + 0.001 * ip, y = 0.5 * frac + 0.002 * ip, z = 0.25 + 0.003 * ip;
        spline2::computeLocationAndFractional(spline_ptr, x - std::floor(x), y - std::floor(y), z - std::floor(z), ix,
                                              iy, iz, a, b, c, da, db, dc, d2a, d2b, d2c);
        RealType* out = scratch_ptr + scratch_size * ip;
        PRAGMA_OFFLOAD("omp parallel for")
        for (size_t sp = 0; sp < padded; sp++)
        {
          spline2offload::evaluate_vgh_impl_v2(spline_ptr, spline_ptr->coefs, ix, iy, iz, static_cast<int>(sp), a, b, c,
                                               da, db, dc, d2a, d2b, d2c, out + sp, padded);
          out[padded * 10 + sp] = out[padded * 4 + sp] * symGGt[0] + out[padded * 5 + sp] * symGGt[1] +
              out[padded * 6 + sp] * symGGt[2] + out[padded * 7 + sp] * symGGt[3] +
              out[padded * 8 + sp] * symGGt[4] + out[padded * 9 + sp] * symGGt[5];
        }
      }
    }
    auto t1old      = std::chrono::steady_clock::now();
    const double ms_old = std::chrono::duration<double, std::milli>(t1old - t0old).count();

    auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < nrepeat; rep++)
    {
      const RealType frac = static_cast<RealType>(rep) / nrepeat;
      PRAGMA_OFFLOAD("omp target teams distribute")
      for (int ip = 0; ip < npos; ip++)
      {
        int ix, iy, iz;
        RealType a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
        const RealType x = frac + 0.001 * ip, y = 0.5 * frac + 0.002 * ip, z = 0.25 + 0.003 * ip;
        spline2::computeLocationAndFractional(spline_ptr, x - std::floor(x), y - std::floor(y), z - std::floor(z), ix,
                                              iy, iz, a, b, c, da, db, dc, d2a, d2b, d2c);
        RealType* out = scratch_ptr + scratch_size * ip;
        PRAGMA_OFFLOAD("omp parallel for")
        for (size_t s = 0; s < padded; s++)
          spline2offload::evaluate_vgl_impl_v2(spline_ptr, spline_ptr->coefs, ix, iy, iz, static_cast<int>(s), a, b, c,
                                               da, db, dc, d2a, d2b, d2c, symGGt, out + s, padded, padded * 10);
      }
    }
    auto t1 = std::chrono::steady_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double per_orbital_ns =
        ms * 1e6 / (static_cast<double>(nrepeat) * npos * padded);
    printf("  %-8d %-12.2f %-12.2f %-14.3f %.2f  ratio_old_over_new=%.3f\n", num_splines, ms_old, ms,
           ms * 1000.0 / (nrepeat * npos), per_orbital_ns, ms_old / ms);

  }
  return 0;
}
