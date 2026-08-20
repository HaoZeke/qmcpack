//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** Compare mapped-device B-spline VGL evaluation against orbital count.
 *
 *  Both paths use one mapped spline object, identical positions and launch geometry,
 *  and separate device-resident output buffers. Numerical validation precedes timing.
 *
 *  Run with the orbital counts to sweep, for example:
 *    OMP_TARGET_OFFLOAD=mandatory benchmark_vgl_offload 32 128 512 513 1024
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <omp.h>
#include <string>
#include <vector>

#include "Configuration.h"
#include "OhmmsPETE/OhmmsVector.h"
#include "OMPTarget/OffloadAlignedAllocators.hpp"
#include "OMPTarget/OMPTargetMath.hpp"
#include "QMCWaveFunctions/BsplineFactory/contraction_helper.hpp"
#include "einspline/bspline_create.h"
#include "spline2/MultiBspline.hpp"
#include "spline2/MultiBsplineOffloadMapper.hpp"
#include "spline2/MultiBsplineVGLH_OMPoffload.hpp"

using namespace qmcplusplus;

namespace
{
using RealType = OHMMS_PRECISION;

constexpr int grid_size           = 20;
constexpr int npos                = 512;
constexpr int nrepeat             = 40;
constexpr int ntrials             = 5;
constexpr size_t ChunkSizePerTeam = 512;
constexpr RealType output_canary  = RealType(-12345.5);

struct SplineSetup
{
  Ugrid grid[3];
  BCtype_d bc[3];
  std::vector<double> data;

  SplineSetup()
  {
    for (int d = 0; d < 3; ++d)
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
    for (int i = 0; i < grid_size; ++i)
      for (int j = 0; j < grid_size; ++j)
        for (int k = 0; k < grid_size; ++k)
        {
          const double x = delta * i;
          const double y = delta * j;
          const double z = delta * k;
          data[(static_cast<size_t>(i) * grid_size + j) * grid_size + k] =
              std::sin(tpi * (x + y)) + 2.0 * std::cos(tpi * (y + z)) + 3.0 * std::sin(tpi * (z + x));
        }
  }
};

bool is_target_device_active()
{
  int initial_device = 1;
  PRAGMA_OFFLOAD("omp target map(from: initial_device)") { initial_device = omp_is_initial_device(); }
  return initial_device == 0;
}

template<typename SplineType>
void evaluate_vgh_trace(const SplineType* spline_ptr,
                        const RealType* GGt_ptr,
                        RealType* output,
                        size_t padded,
                        RealType fraction)
{
  const size_t position_stride = padded * SoAFields3D::NUM_FIELDS;
  const int NumTeams           = static_cast<int>((padded + ChunkSizePerTeam - 1) / ChunkSizePerTeam);
  PRAGMA_OFFLOAD("omp target teams distribute collapse(2) num_teams(NumTeams * npos)")
  for (int ip = 0; ip < npos; ++ip)
    for (int team_id = 0; team_id < NumTeams; ++team_id)
    {
      const size_t first = ChunkSizePerTeam * team_id;
      const size_t last  = omptarget::min(first + ChunkSizePerTeam, padded);
      int ix, iy, iz;
      RealType a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
      const RealType x = fraction + RealType(0.001) * ip;
      const RealType y = RealType(0.5) * fraction + RealType(0.002) * ip;
      const RealType z = RealType(0.25) + RealType(0.003) * ip;
      spline2::computeLocationAndFractional(spline_ptr, x - std::floor(x), y - std::floor(y), z - std::floor(z), ix, iy,
                                            iz, a, b, c, da, db, dc, d2a, d2b, d2c);
      const RealType symGGt[6] = {GGt_ptr[0], GGt_ptr[1] + GGt_ptr[3], GGt_ptr[2] + GGt_ptr[6],
                                  GGt_ptr[4], GGt_ptr[5] + GGt_ptr[7], GGt_ptr[8]};
      RealType* out            = output + position_stride * ip;

      PRAGMA_OFFLOAD("omp parallel for")
      for (int index = 0; index < last - first; ++index)
      {
        const size_t spline_index = first + index;
        spline2offload::evaluate_vgh_impl_v2(spline_ptr, spline_ptr->coefs, ix, iy, iz, static_cast<int>(spline_index),
                                             a, b, c, da, db, dc, d2a, d2b, d2c, out + spline_index, padded);
        out[padded * SoAFields3D::LAPL + spline_index] =
            SymTrace(out[padded * SoAFields3D::HESS00 + spline_index], out[padded * SoAFields3D::HESS01 + spline_index],
                     out[padded * SoAFields3D::HESS02 + spline_index], out[padded * SoAFields3D::HESS11 + spline_index],
                     out[padded * SoAFields3D::HESS12 + spline_index], out[padded * SoAFields3D::HESS22 + spline_index],
                     symGGt);
      }
    }
}

template<typename SplineType>
void evaluate_vgl(const SplineType* spline_ptr,
                  const RealType* GGt_ptr,
                  RealType* output,
                  size_t padded,
                  RealType fraction)
{
  const size_t position_stride = padded * SoAFields3D::NUM_FIELDS;
  const int NumTeams           = static_cast<int>((padded + ChunkSizePerTeam - 1) / ChunkSizePerTeam);
  PRAGMA_OFFLOAD("omp target teams distribute collapse(2) num_teams(NumTeams * npos)")
  for (int ip = 0; ip < npos; ++ip)
    for (int team_id = 0; team_id < NumTeams; ++team_id)
    {
      const size_t first = ChunkSizePerTeam * team_id;
      const size_t last  = omptarget::min(first + ChunkSizePerTeam, padded);
      int ix, iy, iz;
      RealType a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];
      const RealType x = fraction + RealType(0.001) * ip;
      const RealType y = RealType(0.5) * fraction + RealType(0.002) * ip;
      const RealType z = RealType(0.25) + RealType(0.003) * ip;
      spline2::computeLocationAndFractional(spline_ptr, x - std::floor(x), y - std::floor(y), z - std::floor(z), ix, iy,
                                            iz, a, b, c, da, db, dc, d2a, d2b, d2c);
      const RealType symGGt[6] = {GGt_ptr[0], GGt_ptr[1] + GGt_ptr[3], GGt_ptr[2] + GGt_ptr[6],
                                  GGt_ptr[4], GGt_ptr[5] + GGt_ptr[7], GGt_ptr[8]};
      RealType* out            = output + position_stride * ip;

      PRAGMA_OFFLOAD("omp parallel for")
      for (int index = 0; index < last - first; ++index)
      {
        const size_t spline_index = first + index;
        spline2offload::evaluate_vgl_impl_v2(spline_ptr, spline_ptr->coefs, ix, iy, iz, static_cast<int>(spline_index),
                                             a, b, c, da, db, dc, d2a, d2b, d2c, symGGt, out + spline_index, padded,
                                             padded * SoAFields3D::LAPL);
      }
    }
}

template<typename Evaluator>
double measure(Evaluator&& evaluator)
{
  const auto start = std::chrono::steady_clock::now();
  for (int repeat = 0; repeat < nrepeat; ++repeat)
    evaluator(static_cast<RealType>(repeat) / nrepeat);
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

double median(std::vector<double> samples)
{
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

bool validate_outputs(const Vector<RealType, OffloadAllocator<RealType>>& vgh_output,
                      const Vector<RealType, OffloadAllocator<RealType>>& vgl_output,
                      int requested,
                      size_t padded)
{
  const int compared_fields[5] = {SoAFields3D::VAL, SoAFields3D::GRAD0, SoAFields3D::GRAD1, SoAFields3D::GRAD2,
                                  SoAFields3D::LAPL};
  const RealType tolerance     = RealType(256) * std::numeric_limits<RealType>::epsilon();
  const size_t stride          = padded * SoAFields3D::NUM_FIELDS;

  const int mixed_hessian_fields[3] = {SoAFields3D::HESS01, SoAFields3D::HESS02, SoAFields3D::HESS12};
  for (const int field : mixed_hessian_fields)
    if (std::abs(vgh_output[padded * field]) <= RealType(1))
    {
      std::fprintf(stderr, "VGH oracle has a negligible mixed Hessian field: %d\n", field);
      return false;
    }

  for (int ip = 0; ip < npos; ++ip)
    for (const int field : compared_fields)
      for (int spline_index = 0; spline_index < requested; ++spline_index)
      {
        const RealType vgh   = vgh_output[stride * ip + padded * field + spline_index];
        const RealType vgl   = vgl_output[stride * ip + padded * field + spline_index];
        const RealType scale = std::max({RealType(1), std::abs(vgh), std::abs(vgl)});
        if (std::abs(vgh - vgl) > tolerance * scale)
        {
          std::fprintf(stderr,
                       "VGL mismatch: position=%d orbital=%d field=%d vgh_trace=%.17g direct=%.17g tolerance=%.3g\n",
                       ip, spline_index, field, static_cast<double>(vgh), static_cast<double>(vgl),
                       static_cast<double>(tolerance * scale));
          return false;
        }
      }

  const int untouched_fields[6] = {SoAFields3D::HESS00, SoAFields3D::HESS01, SoAFields3D::HESS02,
                                   SoAFields3D::HESS11, SoAFields3D::HESS12, SoAFields3D::HESS22};
  for (int ip = 0; ip < npos; ++ip)
    for (const int field : untouched_fields)
      for (int spline_index = 0; spline_index < requested; ++spline_index)
        if (vgl_output[stride * ip + padded * field + spline_index] != output_canary)
        {
          std::fprintf(stderr, "direct VGL overwrote Hessian storage: position=%d orbital=%d field=%d\n", ip,
                       spline_index, field);
          return false;
        }
  return true;
}

} // namespace

int main(int argc, char** argv)
{
  if (!is_target_device_active())
  {
    std::fprintf(stderr, "benchmark requires execution on an OpenMP target device\n");
    return 3;
  }

  std::vector<int> sizes;
  for (int arg = 1; arg < argc; ++arg)
  {
    const int size = std::atoi(argv[arg]);
    if (size <= 0)
    {
      std::fprintf(stderr, "orbital counts must be positive integers: %s\n", argv[arg]);
      return 1;
    }
    sizes.push_back(size);
  }
  if (sizes.empty())
    sizes = {32, 128, 512, 513, 1024};

  std::printf("# mapped-device B-spline VGL, grid %d^3, %d positions, %d repeats, %d trials\n", grid_size, npos,
              nrepeat, ntrials);
  std::printf("# %-10s %-10s %-12s %-12s %-12s %-12s %s\n", "orbitals", "padded", "vgh_ms", "vgl_ms", "speedup",
              "ns_per_eval", "checksum");

  for (const int requested : sizes)
  {
    SplineSetup setup;
    const size_t padded = getAlignedSize<RealType>(requested);
    MultiBspline<RealType> splines(setup.grid, setup.bc, padded);
    splines.flush_zero();

    UBspline_3d_d* spline = create_UBspline_3d_d(setup.grid[0], setup.grid[1], setup.grid[2], setup.bc[0], setup.bc[1],
                                                 setup.bc[2], setup.data.data());
    if (spline == nullptr)
    {
      std::fprintf(stderr, "failed to construct the reference spline\n");
      return 1;
    }
    for (int index = 0; index < requested; ++index)
      splines.set_spline(*spline, index);
    destroy_Bspline(spline);

    auto* host_spline = splines.getSplinePtr();
    if (host_spline->coefs_size % padded != 0)
    {
      std::fprintf(stderr, "spline coefficient layout is not padded-orbital interleaved\n");
      return 1;
    }
    for (size_t offset = 0; offset < host_spline->coefs_size; offset += padded)
      for (int index = 0; index < requested; ++index)
        host_spline->coefs[offset + index] *= RealType(index + 1);

    MultiBsplineOffloadMapper<RealType> mapper(splines);
    mapper.mapToDevice();
    mapper.updateToDevice();

    // G*G^T for a non-diagonal reciprocal lattice.
    Vector<RealType, OffloadAllocator<RealType>> GGt{RealType(1.05),  RealType(0.20),   RealType(-0.29),
                                                     RealType(0.20),  RealType(0.9025), RealType(0.41),
                                                     RealType(-0.29), RealType(0.41),   RealType(1.26)};
    GGt.updateTo();

    Vector<RealType, OffloadAllocator<RealType>> vgh_output(npos * padded * SoAFields3D::NUM_FIELDS);
    Vector<RealType, OffloadAllocator<RealType>> vgl_output(npos * padded * SoAFields3D::NUM_FIELDS, output_canary);
    vgl_output.updateTo();
    const auto* spline_ptr = splines.getSplinePtr();
    auto* GGt_ptr          = GGt.data();
    auto* vgh_output_ptr   = vgh_output.data();
    auto* vgl_output_ptr   = vgl_output.data();

    const auto run_vgh = [&](RealType fraction) {
      evaluate_vgh_trace(spline_ptr, GGt_ptr, vgh_output_ptr, padded, fraction);
    };
    const auto run_vgl = [&](RealType fraction) {
      evaluate_vgl(spline_ptr, GGt_ptr, vgl_output_ptr, padded, fraction);
    };

    run_vgh(RealType(0.375));
    run_vgl(RealType(0.375));
    vgh_output.updateFrom();
    vgl_output.updateFrom();
    if (!validate_outputs(vgh_output, vgl_output, requested, padded))
      return 2;

    for (int warmup = 0; warmup < 3; ++warmup)
    {
      run_vgh(static_cast<RealType>(warmup) / 7);
      run_vgl(static_cast<RealType>(warmup) / 7);
    }

    std::vector<double> vgh_samples;
    std::vector<double> vgl_samples;
    vgh_samples.reserve(ntrials);
    vgl_samples.reserve(ntrials);
    for (int trial = 0; trial < ntrials; ++trial)
    {
      if (trial % 2 == 0)
      {
        vgh_samples.push_back(measure(run_vgh));
        vgl_samples.push_back(measure(run_vgl));
      }
      else
      {
        vgl_samples.push_back(measure(run_vgl));
        vgh_samples.push_back(measure(run_vgh));
      }
    }

    const double vgh_ms      = median(vgh_samples);
    const double vgl_ms      = median(vgl_samples);
    const double ns_per_eval = vgl_ms * 1e6 / (static_cast<double>(nrepeat) * npos * padded);
    const size_t vgl_stride  = padded * SoAFields3D::NUM_FIELDS;
    double checksum          = 0.0;
    vgl_output.updateFrom();
    for (int ip = 0; ip < npos; ++ip)
      for (int spline_index = 0; spline_index < requested; ++spline_index)
        checksum += static_cast<double>(vgl_output[vgl_stride * ip + spline_index]) +
            static_cast<double>(vgl_output[vgl_stride * ip + padded * SoAFields3D::LAPL + spline_index]);

    std::printf("  %-10d %-10zu %-12.3f %-12.3f %-12.3f %-12.3f %.17g\n", requested, padded, vgh_ms, vgl_ms,
                vgh_ms / vgl_ms, ns_per_eval, checksum);
  }

  return 0;
}
