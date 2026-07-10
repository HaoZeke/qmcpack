//////////////////////////////////////////////////////////////////////////////////////
// Device-friendly free functions for PolynomialFunctor3D evaluation.
// Usable under PRAGMA_OFFLOAD("omp target ...") when ENABLE_OFFLOAD is set;
// when ENABLE_OFFLOAD is unset, PRAGMA_OFFLOAD is empty and this is host code.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_POLYNOMIAL3D_OFFLOAD_H
#define QMCPLUSPLUS_POLYNOMIAL3D_OFFLOAD_H

#include "config.h"

namespace qmcplusplus
{
namespace jeei_offload
{

/** Flat gamma layout matches Array<T,3>: offset = ((l * Lm + m) * Ln + n) */
inline int gamma_index(int l, int m, int n, int Lm, int Ln)
{
  return (l * Lm + m) * Ln + n;
}

/** One triplet VGL; same math as PolynomialFunctor3D::evaluateVGL body. */
template<typename RT>
inline void poly3d_eval_one(const RT r_12,
                            const RT r_1I,
                            const RT r_2I,
                            const RT cutoff_radius,
                            const int N_eI,
                            const int N_ee,
                            const int C,
                            const RT* restrict gamma,
                            RT& val,
                            RT& grad0,
                            RT& grad1,
                            RT& grad2,
                            RT& hess00,
                            RT& hess11,
                            RT& hess22,
                            RT& hess01,
                            RT& hess02)
{
  constexpr RT czero(0);
  constexpr RT cone(1);
  constexpr RT chalf(0.5);
  constexpr RT ctwo(2);
  const RT L  = chalf * cutoff_radius;
  const int Lm = N_eI + 1;
  const int Ln = N_ee + 1;

  val = grad0 = grad1 = grad2 = czero;
  hess00 = hess11 = hess22 = hess01 = hess02 = czero;

  if (r_1I >= L || r_2I >= L)
    return;

  RT r2l(cone), r2l_1(czero), r2l_2(czero), lf(czero);
  for (int l = 0; l <= N_eI; l++)
  {
    RT r2m(cone), r2m_1(czero), r2m_2(czero), mf(czero);
    for (int m = 0; m <= N_eI; m++)
    {
      RT r2n(cone), r2n_1(czero), r2n_2(czero), nf(czero);
      for (int n = 0; n <= N_ee; n++)
      {
        const RT g    = gamma[gamma_index(l, m, n, Lm, Ln)];
        const RT g00x = g * r2l * r2m;
        const RT g10x = g * r2l_1 * r2m;
        const RT g01x = g * r2l * r2m_1;
        const RT gxx0 = g * r2n;
        val += g00x * r2n;
        grad0 += g00x * r2n_1;
        grad1 += g10x * r2n;
        grad2 += g01x * r2n;
        hess00 += g00x * r2n_2;
        hess01 += g10x * r2n_1;
        hess02 += g01x * r2n_1;
        hess11 += gxx0 * r2l_2 * r2m;
        hess22 += gxx0 * r2l * r2m_2;
        nf += cone;
        r2n_2 = r2n_1 * nf;
        r2n_1 = r2n * nf;
        r2n *= r_12;
      }
      mf += cone;
      r2m_2 = r2m_1 * mf;
      r2m_1 = r2m * mf;
      r2m *= r_2I;
    }
    lf += cone;
    r2l_2 = r2l_1 * lf;
    r2l_1 = r2l * lf;
    r2l *= r_1I;
  }

  const RT r_2I_minus_L = r_2I - L;
  const RT r_1I_minus_L = r_1I - L;
  const RT both_minus_L = r_2I_minus_L * r_1I_minus_L;
  for (int i = 0; i < C; i++)
  {
    hess00 = both_minus_L * hess00;
    hess01 = both_minus_L * hess01 + r_2I_minus_L * grad0;
    hess02 = both_minus_L * hess02 + r_1I_minus_L * grad0;
    hess11 = both_minus_L * hess11 + ctwo * r_2I_minus_L * grad1;
    hess22 = both_minus_L * hess22 + ctwo * r_1I_minus_L * grad2;
    grad0  = both_minus_L * grad0;
    grad1  = both_minus_L * grad1 + r_2I_minus_L * val;
    grad2  = both_minus_L * grad2 + r_1I_minus_L * val;
    val *= both_minus_L;
  }
  // match evaluateVGL scaling
  grad0 /= r_12;
  grad1 /= r_1I;
  grad2 /= r_2I;
  hess01 /= (r_12 * r_1I);
  hess02 /= (r_12 * r_2I);
}

/** Dense dual-table VGL for moving electron jel.
 *  Layout:
 *    ee_r   [Ne_pad], ee_dr[3][Ne_pad]
 *    ei_r_j [Ni_pad], ei_dr_j[3][Ni_pad]  -- temp row for jel
 *    ei_r_full [Nelec * Ni_pad], ei_dr_full [3 * Nelec * Ni_pad] row-major kel-major
 *  Functor tables size nfun = iGroups * eGroups * eGroups:
 *    gamma_ptrs[nfun], cutoffs[nfun], N_eI[nfun], N_ee[nfun], C[nfun]  (null gamma => skip)
 *  Outputs:
 *    Uj, dUj[3], d2Uj
 *    Uk[Nelec], dUk[3*Nelec_pad], d2Uk[Nelec]  (partner contributions)
 */
template<typename RT>
inline void dense_ratio_grad_one_walker(const int jel,
                                        const int Nelec,
                                        const int Ne_pad,
                                        const int Nion,
                                        const int Ni_pad,
                                        const int eGroups,
                                        const int iGroups,
                                        const int* restrict e_grp,   // [Nelec]
                                        const int* restrict i_grp,   // [Nion]
                                        const RT* restrict ion_cut,  // [Nion]
                                        const RT* restrict ee_r,
                                        const RT* restrict ee_dr, // [3][Ne_pad]
                                        const RT* restrict ei_r_j,
                                        const RT* restrict ei_dr_j, // [3][Ni_pad]
                                        const RT* restrict ei_r_full,  // [kel * Ni_pad + iat]
                                        const RT* restrict ei_dr_full, // [idim * Nelec * Ni_pad + kel * Ni_pad + iat]
                                        const RT** gamma_ptrs,
                                        const RT* restrict fun_cutoff,
                                        const int* restrict fun_NeI,
                                        const int* restrict fun_Nee,
                                        const int* restrict fun_C,
                                        RT& Uj,
                                        RT* restrict dUj, // [3]
                                        RT& d2Uj,
                                        RT* restrict Uk,
                                        RT* restrict dUk, // [3][Ne_pad]
                                        RT* restrict d2Uk)
{
  constexpr RT czero(0);
  constexpr RT cone(1);
  constexpr RT ctwo(2);
  constexpr RT lapfac = RT(3) - cone; // OHMMS_DIM=3

  Uj   = czero;
  d2Uj = czero;
  dUj[0] = dUj[1] = dUj[2] = czero;
  for (int k = 0; k < Nelec; ++k)
  {
    Uk[k]   = czero;
    d2Uk[k] = czero;
  }
  for (int idim = 0; idim < 3; ++idim)
    for (int k = 0; k < Ne_pad; ++k)
      dUk[idim * Ne_pad + k] = czero;

  const int jg = e_grp[jel];

  for (int iat = 0; iat < Nion; ++iat)
  {
    const RT r_jI = ei_r_j[iat];
    if (r_jI >= ion_cut[iat])
      continue;
    const int ig = i_grp[iat];
    const RT jIx = ei_dr_j[0 * Ni_pad + iat];
    const RT jIy = ei_dr_j[1 * Ni_pad + iat];
    const RT jIz = ei_dr_j[2 * Ni_pad + iat];

    for (int kel = 0; kel < Nelec; ++kel)
    {
      if (kel == jel)
        continue;
      const RT r_kI = ei_r_full[kel * Ni_pad + iat];
      if (r_kI >= ion_cut[iat])
        continue;
      const int kg  = e_grp[kel];
      const int fid = (ig * eGroups + jg) * eGroups + kg;
      const RT* gamma = gamma_ptrs[fid];
      if (gamma == nullptr)
        continue;

      const RT r_jk = ee_r[kel];
      const RT jkx  = ee_dr[0 * Ne_pad + kel];
      const RT jky  = ee_dr[1 * Ne_pad + kel];
      const RT jkz  = ee_dr[2 * Ne_pad + kel];
      const RT kIx  = ei_dr_full[0 * Nelec * Ni_pad + kel * Ni_pad + iat];
      const RT kIy  = ei_dr_full[1 * Nelec * Ni_pad + kel * Ni_pad + iat];
      const RT kIz  = ei_dr_full[2 * Nelec * Ni_pad + kel * Ni_pad + iat];

      RT val, g0, g1, g2, h00, h11, h22, h01, h02;
      poly3d_eval_one(r_jk, r_jI, r_kI, fun_cutoff[fid], fun_NeI[fid], fun_Nee[fid], fun_C[fid], gamma, val, g0, g1,
                      g2, h00, h11, h22, h01, h02);

      Uj += val;
      d2Uj -= h00 + h11 + lapfac * (g0 + g1);

      // dUj from jk and jI channels (same as computeU3_engine)
      dUj[0] += g1 * jIx + g0 * jkx;
      dUj[1] += g1 * jIy + g0 * jky;
      dUj[2] += g1 * jIz + g0 * jkz;

      // cross term in lap of j (hess01 * jk·jI style)
      const RT jk_dot_jI = jkx * jIx + jky * jIy + jkz * jIz;
      d2Uj -= ctwo * h01 * jk_dot_jI;

      // partner kel
      Uk[kel] += val;
      const RT k_contrib_x = kIx * g2 - jkx * g0;
      const RT k_contrib_y = kIy * g2 - jky * g0;
      const RT k_contrib_z = kIz * g2 - jkz * g0;
      dUk[0 * Ne_pad + kel] += k_contrib_x;
      dUk[1 * Ne_pad + kel] += k_contrib_y;
      dUk[2 * Ne_pad + kel] += k_contrib_z;

      const RT kI_dot_jk = kIx * jkx + kIy * jky + kIz * jkz;
      d2Uk[kel] -= h00 + h22 + lapfac * (g0 + g2) - ctwo * h02 * kI_dot_jk;
    }
  }
}

} // namespace jeei_offload
} // namespace qmcplusplus
#endif
