//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2023 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#include "OMPTarget/OMPTargetMath.hpp"

namespace qmcplusplus
{
namespace C2C
{
enum CompactVGLFields
{
  VGL_VALUE = 0,
  VGL_GRAD0,
  VGL_GRAD1,
  VGL_GRAD2,
  VGL_LAPL,
  VGL_NUM_FIELDS
};

inline void complex_index_bounds(size_t first_real,
                                 size_t last_real,
                                 size_t num_orbitals,
                                 size_t& first_complex,
                                 size_t& last_complex)
{
  first_complex = first_real / 2;
  last_complex  = omptarget::min(last_real / 2, num_orbitals);
}

template<typename ST>
inline void phase_sincos(ST x, ST y, ST z, ST kX, ST kY, ST kZ, ST& s, ST& c)
{
  const ST phase_xy = std::fma(y, kY, x * kX);
  const ST phase    = -std::fma(z, kZ, phase_xy);
  omptarget::sincos(phase, &s, &c);
}

template<typename ST, typename TT>
inline TT apply_phase(ST s, ST c, ST value_r, ST value_i)
{
  return TT(std::fma(-s, value_i, c * value_r), std::fma(s, value_r, c * value_i));
}

template<typename ST, typename TT>
inline TT apply_phase_value(ST x, ST y, ST z, ST value_r, ST value_i, ST kX, ST kY, ST kZ)
{
  ST s;
  ST c;
  phase_sincos(x, y, z, kX, kY, kZ, s, c);
  return apply_phase<ST, TT>(s, c, value_r, value_i);
}

template<typename ST, typename TT>
inline void apply_phase_vgl(ST x,
                            ST y,
                            ST z,
                            ST val_r,
                            ST val_i,
                            ST g0_r,
                            ST g0_i,
                            ST g1_r,
                            ST g1_i,
                            ST g2_r,
                            ST g2_i,
                            ST lcart_r,
                            ST lcart_i,
                            const ST G[9],
                            ST kX,
                            ST kY,
                            ST kZ,
                            ST mKK,
                            TT& psi,
                            TT& dpsi_x,
                            TT& dpsi_y,
                            TT& dpsi_z,
                            TT& d2psi)
{
  constexpr ST two(2);
  const ST &g00 = G[0], &g01 = G[1], &g02 = G[2], &g10 = G[3], &g11 = G[4], &g12 = G[5], &g20 = G[6], &g21 = G[7],
           &g22 = G[8];

  ST s;
  ST c;
  phase_sincos(x, y, z, kX, kY, kZ, s, c);

  const ST dX_r = g00 * g0_r + g01 * g1_r + g02 * g2_r;
  const ST dY_r = g10 * g0_r + g11 * g1_r + g12 * g2_r;
  const ST dZ_r = g20 * g0_r + g21 * g1_r + g22 * g2_r;

  const ST dX_i = g00 * g0_i + g01 * g1_i + g02 * g2_i;
  const ST dY_i = g10 * g0_i + g11 * g1_i + g12 * g2_i;
  const ST dZ_i = g20 * g0_i + g21 * g1_i + g22 * g2_i;

  const ST gX_r = dX_r + val_i * kX;
  const ST gY_r = dY_r + val_i * kY;
  const ST gZ_r = dZ_r + val_i * kZ;
  const ST gX_i = dX_i - val_r * kX;
  const ST gY_i = dY_i - val_r * kY;
  const ST gZ_i = dZ_i - val_r * kZ;

  const ST lap_r = lcart_r + mKK * val_r + two * (kX * dX_i + kY * dY_i + kZ * dZ_i);
  const ST lap_i = lcart_i + mKK * val_i - two * (kX * dX_r + kY * dY_r + kZ * dZ_r);

  psi     = apply_phase<ST, TT>(s, c, val_r, val_i);
  d2psi   = apply_phase<ST, TT>(s, c, lap_r, lap_i);
  dpsi_x  = apply_phase<ST, TT>(s, c, gX_r, gX_i);
  dpsi_y  = apply_phase<ST, TT>(s, c, gY_r, gY_i);
  dpsi_z  = apply_phase<ST, TT>(s, c, gZ_r, gZ_i);
}

template<typename ST, typename TT>
inline void assign_v(ST x,
                     ST y,
                     ST z,
                     TT* restrict results_scratch_ptr,
                     const ST* restrict offload_scratch_ptr,
                     const ST* restrict myKcart_ptr,
                     size_t myKcart_padded_size,
                     int index)
{
  const ST* restrict kx = myKcart_ptr;
  const ST* restrict ky = myKcart_ptr + myKcart_padded_size;
  const ST* restrict kz = myKcart_ptr + myKcart_padded_size * 2;

  const ST* restrict val = offload_scratch_ptr;
  TT* restrict psi       = results_scratch_ptr;

  const ST val_r = val[index * 2];
  const ST val_i = val[index * 2 + 1];
  psi[index] = apply_phase_value<ST, TT>(x, y, z, val_r, val_i, kx[index], ky[index], kz[index]);
}

/** assign_vgl
   */
template<typename ST, typename TT>
inline void assign_vgl(ST x,
                       ST y,
                       ST z,
                       TT* restrict results_scratch_ptr,
                       size_t orb_padded_size,
                       const ST* mKK_ptr,
                       const ST* restrict offload_scratch_ptr,
                       size_t spline_padded_size,
                       const ST G[9],
                       const ST* myKcart_ptr,
                       size_t myKcart_padded_size,
                       int index)
{
  const ST* restrict k0 = myKcart_ptr;
  const ST* restrict k1 = myKcart_ptr + myKcart_padded_size;
  const ST* restrict k2 = myKcart_ptr + myKcart_padded_size * 2;

  const ST* restrict val   = offload_scratch_ptr + spline_padded_size * SoAFields3D::VAL;
  const ST* restrict g0    = offload_scratch_ptr + spline_padded_size * SoAFields3D::GRAD0;
  const ST* restrict g1    = offload_scratch_ptr + spline_padded_size * SoAFields3D::GRAD1;
  const ST* restrict g2    = offload_scratch_ptr + spline_padded_size * SoAFields3D::GRAD2;
  const ST* restrict lcart = offload_scratch_ptr + spline_padded_size * SoAFields3D::LAPL;

  const size_t jr = index << 1;
  const size_t ji = jr + 1;

  const ST kX    = k0[index];
  const ST kY    = k1[index];
  const ST kZ    = k2[index];
  const ST val_r = val[jr];
  const ST val_i = val[ji];

  TT* restrict psi    = results_scratch_ptr;
  TT* restrict dpsi_x = results_scratch_ptr + orb_padded_size;
  TT* restrict dpsi_y = results_scratch_ptr + orb_padded_size * 2;
  TT* restrict dpsi_z = results_scratch_ptr + orb_padded_size * 3;
  TT* restrict d2psi  = results_scratch_ptr + orb_padded_size * 4;

  apply_phase_vgl(x, y, z, val_r, val_i, g0[jr], g0[ji], g1[jr], g1[ji], g2[jr], g2[ji], lcart[jr],
                  lcart[ji], G, kX, kY, kZ, mKK_ptr[index], psi[index], dpsi_x[index], dpsi_y[index],
                  dpsi_z[index], d2psi[index]);
}
} // namespace C2C
} // namespace qmcplusplus
