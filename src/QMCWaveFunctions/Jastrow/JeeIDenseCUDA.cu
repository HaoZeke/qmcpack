#include "JeeIDenseCUDA.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace qmcplusplus
{
namespace jeei_cuda
{
namespace
{

inline void check(cudaError_t st, const char* what)
{
  if (st != cudaSuccess)
    throw std::runtime_error(std::string("JeeIDenseCUDA ") + what + ": " + cudaGetErrorString(st));
}

__device__ inline int gamma_index(int l, int m, int n, int Lm, int Ln)
{
  return (l * Lm + m) * Ln + n;
}

__device__ void poly3d_eval_one(const double r_12,
                               const double r_1I,
                               const double r_2I,
                               const double cutoff_radius,
                               const int N_eI,
                               const int N_ee,
                               const int C,
                               const double* gamma,
                               double& val,
                               double& grad0,
                               double& grad1,
                               double& grad2,
                               double& hess00,
                               double& hess11,
                               double& hess22,
                               double& hess01,
                               double& hess02)
{
  const double czero = 0.0;
  const double cone  = 1.0;
  const double chalf = 0.5;
  const double ctwo  = 2.0;
  const double L     = chalf * cutoff_radius;
  const int Lm       = N_eI + 1;
  const int Ln       = N_ee + 1;

  val = grad0 = grad1 = grad2 = czero;
  hess00 = hess11 = hess22 = hess01 = hess02 = czero;
  if (r_1I >= L || r_2I >= L)
    return;

  double r2l = cone, r2l_1 = czero, r2l_2 = czero, lf = czero;
  for (int l = 0; l <= N_eI; l++)
  {
    double r2m = cone, r2m_1 = czero, r2m_2 = czero, mf = czero;
    for (int m = 0; m <= N_eI; m++)
    {
      double r2n = cone, r2n_1 = czero, r2n_2 = czero, nf = czero;
      for (int n = 0; n <= N_ee; n++)
      {
        const double g    = gamma[gamma_index(l, m, n, Lm, Ln)];
        const double g00x = g * r2l * r2m;
        const double g10x = g * r2l_1 * r2m;
        const double g01x = g * r2l * r2m_1;
        const double gxx0 = g * r2n;
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

  const double r_2I_minus_L = r_2I - L;
  const double r_1I_minus_L = r_1I - L;
  const double both_minus_L = r_2I_minus_L * r_1I_minus_L;
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
  // Chain-rule divisions as in the host evaluateVGL, guarded per denominator:
  // physical pair distances are strictly positive, so the zero branches only fire
  // on degenerate inputs and keep the outputs finite instead of 0/0.
  const double eps = 1e-12;
  if (r_12 > eps)
    grad0 /= r_12;
  else
    grad0 = 0.0;
  if (r_1I > eps)
    grad1 /= r_1I;
  else
    grad1 = 0.0;
  if (r_2I > eps)
    grad2 /= r_2I;
  else
    grad2 = 0.0;
  if (r_12 > eps && r_1I > eps)
    hess01 /= (r_12 * r_1I);
  else
    hess01 = 0.0;
  if (r_12 > eps && r_2I > eps)
    hess02 /= (r_12 * r_2I);
  else
    hess02 = 0.0;
}

// Cut1: ratioGrad — one block per walker, threads own kel
__global__ void dense_ratio_grad_kernel(const int jel,
                                        const int nw,
                                        const int Nelec,
                                        const int Ne_pad,
                                        const int Nion,
                                        const int Ni_pad,
                                        const int eGroups,
                                        const int nfun,
                                        const double* __restrict__ ee_temp,
                                        const double* __restrict__ ei_temp,
                                        const double* __restrict__ ei_full_r,
                                        const double* __restrict__ ei_full_dr,
                                        const int* __restrict__ e_grp,
                                        const int* __restrict__ i_grp,
                                        const double* __restrict__ ion_cut,
                                        const double* __restrict__ gamma_pool,
                                        const int* __restrict__ gamma_offset,
                                        const double* __restrict__ fun_cut,
                                        const int* __restrict__ fun_NeI,
                                        const int* __restrict__ fun_Nee,
                                        const int* __restrict__ fun_C,
                                        const double* __restrict__ Uat_state,
                                        double* __restrict__ vgl,
                                        double* __restrict__ Uk,
                                        double* __restrict__ dUk,
                                        double* __restrict__ d2Uk)
{
  const int iw = blockIdx.x;
  if (iw >= nw)
    return;

  const int tid     = threadIdx.x;
  const int nthr    = blockDim.x;
  const double lapfac = 2.0;

  const double* ee_base = ee_temp + iw * 4 * Ne_pad;
  const double* ei_base = ei_temp + iw * 4 * Ni_pad;
  const double* fr      = ei_full_r + iw * Nelec * Ni_pad;
  const double* fdr     = ei_full_dr + iw * 3 * Nelec * Ni_pad;

  double* Uk_w  = Uk + iw * Ne_pad;
  double* d2_w  = d2Uk + iw * Ne_pad;
  double* dUk_w = dUk + iw * 3 * Ne_pad;

  const int jg = e_grp[jel];

  double Uj = 0.0, d2Uj = 0.0;
  double dUj0 = 0.0, dUj1 = 0.0, dUj2 = 0.0;

  for (int kel = tid; kel < Nelec; kel += nthr)
  {
    double uk = 0.0, d2k = 0.0;
    double duk0 = 0.0, duk1 = 0.0, duk2 = 0.0;

    if (kel != jel)
    {
      const double r_jk = ee_base[kel];
      const double jkx  = ee_base[1 * Ne_pad + kel];
      const double jky  = ee_base[2 * Ne_pad + kel];
      const double jkz  = ee_base[3 * Ne_pad + kel];
      const int kg      = e_grp[kel];

      for (int iat = 0; iat < Nion; ++iat)
      {
        const double r_jI = ei_base[iat];
        if (r_jI >= ion_cut[iat])
          continue;
        const double r_kI = fr[kel * Ni_pad + iat];
        if (r_kI >= ion_cut[iat])
          continue;

        const int ig  = i_grp[iat];
        const int fid = (ig * eGroups + jg) * eGroups + kg;
        if (fid < 0 || fid >= nfun || gamma_offset[fid] < 0)
          continue;

        const double jIx = ei_base[1 * Ni_pad + iat];
        const double jIy = ei_base[2 * Ni_pad + iat];
        const double jIz = ei_base[3 * Ni_pad + iat];
        const double kIx = fdr[0 * Nelec * Ni_pad + kel * Ni_pad + iat];
        const double kIy = fdr[1 * Nelec * Ni_pad + kel * Ni_pad + iat];
        const double kIz = fdr[2 * Nelec * Ni_pad + kel * Ni_pad + iat];

        double val, g0, g1, g2, h00, h11, h22, h01, h02;
        poly3d_eval_one(r_jk, r_jI, r_kI, fun_cut[fid], fun_NeI[fid], fun_Nee[fid], fun_C[fid],
                        gamma_pool + gamma_offset[fid], val, g0, g1, g2, h00, h11, h22, h01, h02);

        Uj += val;
        d2Uj -= h00 + h11 + lapfac * (g0 + g1);
        dUj0 += g1 * jIx + g0 * jkx;
        dUj1 += g1 * jIy + g0 * jky;
        dUj2 += g1 * jIz + g0 * jkz;
        d2Uj -= 2.0 * h01 * (jkx * jIx + jky * jIy + jkz * jIz);

        uk += val;
        duk0 += kIx * g2 - jkx * g0;
        duk1 += kIy * g2 - jky * g0;
        duk2 += kIz * g2 - jkz * g0;
        d2k -= h00 + h22 + lapfac * (g0 + g2) - 2.0 * h02 * (kIx * jkx + kIy * jky + kIz * jkz);
      }
    }

    Uk_w[kel]               = uk;
    d2_w[kel]               = d2k;
    dUk_w[0 * Ne_pad + kel] = duk0;
    dUk_w[1 * Ne_pad + kel] = duk1;
    dUk_w[2 * Ne_pad + kel] = duk2;
  }

  __shared__ double sh[5 * 256];
  sh[tid]        = Uj;
  sh[256 + tid]  = dUj0;
  sh[512 + tid]  = dUj1;
  sh[768 + tid]  = dUj2;
  sh[1024 + tid] = d2Uj;
  __syncthreads();
  for (int s = nthr / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      sh[tid] += sh[tid + s];
      sh[256 + tid] += sh[256 + tid + s];
      sh[512 + tid] += sh[512 + tid + s];
      sh[768 + tid] += sh[768 + tid + s];
      sh[1024 + tid] += sh[1024 + tid + s];
    }
    __syncthreads();
  }
  if (tid == 0)
  {
    double* v = vgl + iw * 6;
    v[0] = sh[0];
    v[1] = sh[256];
    v[2] = sh[512];
    v[3] = sh[768];
    v[4] = sh[1024];
    // pre-accept Uat[jel] from the resident state: the acceptance ratio must see
    // partner updates from earlier accepts in this sweep, which live device-side
    v[5] = Uat_state[iw * Ne_pad + jel];
  }
}

// Cut2: recompute — triangle assemble matching host recompute(computeU3 triangle)
// ee_full_r layout: [nw][Nelec][Ne_pad] lower triangle only (kel < jel)
// ee_full_dr: [nw][3][Nelec][Ne_pad]
// ei_full same as ratioGrad
__global__ void dense_recompute_kernel(const int nw,
                                       const int Nelec,
                                       const int Ne_pad,
                                       const int Nion,
                                       const int Ni_pad,
                                       const int eGroups,
                                       const int nfun,
                                       const double* __restrict__ ee_full_r,
                                       const double* __restrict__ ee_full_dr,
                                       const double* __restrict__ ei_full_r,
                                       const double* __restrict__ ei_full_dr,
                                       const int* __restrict__ e_grp,
                                       const int* __restrict__ i_grp,
                                       const double* __restrict__ ion_cut,
                                       const double* __restrict__ gamma_pool,
                                       const int* __restrict__ gamma_offset,
                                       const double* __restrict__ fun_cut,
                                       const int* __restrict__ fun_NeI,
                                       const int* __restrict__ fun_Nee,
                                       const int* __restrict__ fun_C,
                                       double* __restrict__ Uat,
                                       double* __restrict__ dUat,
                                       double* __restrict__ d2Uat)
{
  const int iw = blockIdx.x;
  if (iw >= nw)
    return;

  const int tid  = threadIdx.x;
  const int nthr = blockDim.x;
  const double lapfac = 2.0;

  double* U_w   = Uat + iw * Ne_pad;
  double* d2_w  = d2Uat + iw * Ne_pad;
  double* dU_w  = dUat + iw * 3 * Ne_pad;

  for (int e = tid; e < Nelec; e += nthr)
  {
    U_w[e]               = 0.0;
    d2_w[e]              = 0.0;
    dU_w[0 * Ne_pad + e] = 0.0;
    dU_w[1 * Ne_pad + e] = 0.0;
    dU_w[2 * Ne_pad + e] = 0.0;
  }
  __syncthreads();

  const double* fr  = ei_full_r + iw * Nelec * Ni_pad;
  const double* fdr = ei_full_dr + iw * 3 * Nelec * Ni_pad;
  const double* eer = ee_full_r + iw * Nelec * Ne_pad;
  const double* eed = ee_full_dr + iw * 3 * Nelec * Ne_pad;

  for (int jel = 0; jel < Nelec; ++jel)
  {
    const int jg = e_grp[jel];
    // ei row for jel
    const double* ei_r  = fr + jel * Ni_pad;
    const double* ei_dx = fdr + 0 * Nelec * Ni_pad + jel * Ni_pad;
    const double* ei_dy = fdr + 1 * Nelec * Ni_pad + jel * Ni_pad;
    const double* ei_dz = fdr + 2 * Nelec * Ni_pad + jel * Ni_pad;
    // ee lower row
    const double* ee_r  = eer + jel * Ne_pad;
    const double* ee_dx = eed + 0 * Nelec * Ne_pad + jel * Ne_pad;
    const double* ee_dy = eed + 1 * Nelec * Ne_pad + jel * Ne_pad;
    const double* ee_dz = eed + 2 * Nelec * Ne_pad + jel * Ne_pad;

    double Uj = 0.0, d2Uj = 0.0;
    double dUj0 = 0.0, dUj1 = 0.0, dUj2 = 0.0;

    // triangle: kel < jel only
    for (int kel = tid; kel < jel; kel += nthr)
    {
      double uk = 0.0, d2k = 0.0;
      double duk0 = 0.0, duk1 = 0.0, duk2 = 0.0;

      const double r_jk = ee_r[kel];
      const double jkx  = ee_dx[kel];
      const double jky  = ee_dy[kel];
      const double jkz  = ee_dz[kel];
      const int kg      = e_grp[kel];

      for (int iat = 0; iat < Nion; ++iat)
      {
        const double r_jI = ei_r[iat];
        if (r_jI >= ion_cut[iat])
          continue;
        const double r_kI = fr[kel * Ni_pad + iat];
        if (r_kI >= ion_cut[iat])
          continue;

        const int ig  = i_grp[iat];
        const int fid = (ig * eGroups + jg) * eGroups + kg;
        if (fid < 0 || fid >= nfun || gamma_offset[fid] < 0)
          continue;

        const double jIx = ei_dx[iat];
        const double jIy = ei_dy[iat];
        const double jIz = ei_dz[iat];
        const double kIx = fdr[0 * Nelec * Ni_pad + kel * Ni_pad + iat];
        const double kIy = fdr[1 * Nelec * Ni_pad + kel * Ni_pad + iat];
        const double kIz = fdr[2 * Nelec * Ni_pad + kel * Ni_pad + iat];

        double val, g0, g1, g2, h00, h11, h22, h01, h02;
        poly3d_eval_one(r_jk, r_jI, r_kI, fun_cut[fid], fun_NeI[fid], fun_Nee[fid], fun_C[fid],
                        gamma_pool + gamma_offset[fid], val, g0, g1, g2, h00, h11, h22, h01, h02);

        Uj += val;
        d2Uj -= h00 + h11 + lapfac * (g0 + g1);
        dUj0 += g1 * jIx + g0 * jkx;
        dUj1 += g1 * jIy + g0 * jky;
        dUj2 += g1 * jIz + g0 * jkz;
        d2Uj -= 2.0 * h01 * (jkx * jIx + jky * jIy + jkz * jIz);

        uk += val;
        duk0 += kIx * g2 - jkx * g0;
        duk1 += kIy * g2 - jky * g0;
        duk2 += kIz * g2 - jkz * g0;
        d2k -= h00 + h22 + lapfac * (g0 + g2) - 2.0 * h02 * (kIx * jkx + kIy * jky + kIz * jkz);
      }

      // scatter to lower triangle partners
      atomicAdd(&U_w[kel], uk);
      atomicAdd(&d2_w[kel], d2k);
      atomicAdd(&dU_w[0 * Ne_pad + kel], duk0);
      atomicAdd(&dU_w[1 * Ne_pad + kel], duk1);
      atomicAdd(&dU_w[2 * Ne_pad + kel], duk2);
    }

    __shared__ double sh[5 * 256];
    sh[tid]        = Uj;
    sh[256 + tid]  = dUj0;
    sh[512 + tid]  = dUj1;
    sh[768 + tid]  = dUj2;
    sh[1024 + tid] = d2Uj;
    __syncthreads();
    for (int s = nthr / 2; s > 0; s >>= 1)
    {
      if (tid < s)
      {
        sh[tid] += sh[tid + s];
        sh[256 + tid] += sh[256 + tid + s];
        sh[512 + tid] += sh[512 + tid + s];
        sh[768 + tid] += sh[768 + tid + s];
        sh[1024 + tid] += sh[1024 + tid + s];
      }
      __syncthreads();
    }
    if (tid == 0)
    {
      U_w[jel]               = sh[0];
      dU_w[0 * Ne_pad + jel] = sh[256];
      dU_w[1 * Ne_pad + jel] = sh[512];
      dU_w[2 * Ne_pad + jel] = sh[768];
      d2_w[jel]              = sh[1024];
    }
    __syncthreads();
  }
}

// Device accept: old-side dense pass from resident full tables, delta-apply into
// resident Uat state, then patch the moved electron's ee/ei rows from the resident
// temp buffers (uploaded by the preceding ratioGrad on the same stream).
__global__ void dense_accept_kernel(const int jel,
                                    const int nw,
                                    const int Nelec,
                                    const int Ne_pad,
                                    const int Nion,
                                    const int Ni_pad,
                                    const int eGroups,
                                    const int nfun,
                                    const int* __restrict__ accepted,
                                    const double* __restrict__ ee_temp,
                                    const double* __restrict__ ei_temp,
                                    double* __restrict__ ee_full_r,
                                    double* __restrict__ ee_full_dr,
                                    double* __restrict__ ei_full_r,
                                    double* __restrict__ ei_full_dr,
                                    const int* __restrict__ e_grp,
                                    const int* __restrict__ i_grp,
                                    const double* __restrict__ ion_cut,
                                    const double* __restrict__ gamma_pool,
                                    const int* __restrict__ gamma_offset,
                                    const double* __restrict__ fun_cut,
                                    const int* __restrict__ fun_NeI,
                                    const int* __restrict__ fun_Nee,
                                    const int* __restrict__ fun_C,
                                    const double* __restrict__ vgl,
                                    const double* __restrict__ Uk,
                                    const double* __restrict__ dUk,
                                    const double* __restrict__ d2Uk,
                                    double* __restrict__ Uat,
                                    double* __restrict__ dUat,
                                    double* __restrict__ d2Uat)
{
  const int iw = blockIdx.x;
  if (iw >= nw || !accepted[iw])
    return;

  const int tid       = threadIdx.x;
  const int nthr      = blockDim.x;
  const double lapfac = 2.0;

  const double* fr  = ei_full_r + size_t(iw) * Nelec * Ni_pad;
  const double* fdr = ei_full_dr + size_t(iw) * 3 * Nelec * Ni_pad;
  double* eer       = ee_full_r + size_t(iw) * Nelec * Ne_pad;
  double* eed       = ee_full_dr + size_t(iw) * 3 * Nelec * Ne_pad;

  const double* Uk_w  = Uk + size_t(iw) * Ne_pad;
  const double* d2_w  = d2Uk + size_t(iw) * Ne_pad;
  const double* dUk_w = dUk + size_t(iw) * 3 * Ne_pad;
  double* U_s         = Uat + size_t(iw) * Ne_pad;
  double* d2_s        = d2Uat + size_t(iw) * Ne_pad;
  double* dU_s        = dUat + size_t(iw) * 3 * Ne_pad;

  const int jg          = e_grp[jel];
  const size_t plane_ee = size_t(Nelec) * Ne_pad;
  const size_t plane_ei = size_t(Nelec) * Ni_pad;

  // Phase 1: old-side contributions of jel to each partner from resident tables,
  // applied together with the resident new-side Uk buffers.
  for (int kel = tid; kel < Nelec; kel += nthr)
  {
    double uk = 0.0, d2k = 0.0;
    double duk0 = 0.0, duk1 = 0.0, duk2 = 0.0;

    if (kel != jel)
    {
      double r_jk, jkx, jky, jkz;
      if (kel < jel)
      {
        r_jk = eer[size_t(jel) * Ne_pad + kel];
        jkx  = eed[0 * plane_ee + size_t(jel) * Ne_pad + kel];
        jky  = eed[1 * plane_ee + size_t(jel) * Ne_pad + kel];
        jkz  = eed[2 * plane_ee + size_t(jel) * Ne_pad + kel];
      }
      else
      {
        // pair stored in row kel, column jel with roles swapped: displacement flips sign
        r_jk = eer[size_t(kel) * Ne_pad + jel];
        jkx  = -eed[0 * plane_ee + size_t(kel) * Ne_pad + jel];
        jky  = -eed[1 * plane_ee + size_t(kel) * Ne_pad + jel];
        jkz  = -eed[2 * plane_ee + size_t(kel) * Ne_pad + jel];
      }
      const int kg = e_grp[kel];

      for (int iat = 0; iat < Nion; ++iat)
      {
        const double r_jI = fr[size_t(jel) * Ni_pad + iat];
        if (r_jI >= ion_cut[iat])
          continue;
        const double r_kI = fr[size_t(kel) * Ni_pad + iat];
        if (r_kI >= ion_cut[iat])
          continue;

        const int ig  = i_grp[iat];
        const int fid = (ig * eGroups + jg) * eGroups + kg;
        if (fid < 0 || fid >= nfun || gamma_offset[fid] < 0)
          continue;

        const double kIx = fdr[0 * plane_ei + size_t(kel) * Ni_pad + iat];
        const double kIy = fdr[1 * plane_ei + size_t(kel) * Ni_pad + iat];
        const double kIz = fdr[2 * plane_ei + size_t(kel) * Ni_pad + iat];

        double val, g0, g1, g2, h00, h11, h22, h01, h02;
        poly3d_eval_one(r_jk, r_jI, r_kI, fun_cut[fid], fun_NeI[fid], fun_Nee[fid], fun_C[fid],
                        gamma_pool + gamma_offset[fid], val, g0, g1, g2, h00, h11, h22, h01, h02);

        uk += val;
        duk0 += kIx * g2 - jkx * g0;
        duk1 += kIy * g2 - jky * g0;
        duk2 += kIz * g2 - jkz * g0;
        d2k -= h00 + h22 + lapfac * (g0 + g2) - 2.0 * h02 * (kIx * jkx + kIy * jky + kIz * jkz);
      }
    }

    U_s[kel] += Uk_w[kel] - uk;
    d2_s[kel] += d2_w[kel] - d2k;
    dU_s[0 * Ne_pad + kel] += dUk_w[0 * Ne_pad + kel] - duk0;
    dU_s[1 * Ne_pad + kel] += dUk_w[1 * Ne_pad + kel] - duk1;
    dU_s[2 * Ne_pad + kel] += dUk_w[2 * Ne_pad + kel] - duk2;
  }
  __syncthreads();

  // Phase 2: moved electron takes the batched-VGL values; patch resident rows.
  if (tid == 0)
  {
    const double* v      = vgl + size_t(iw) * 6;
    U_s[jel]             = v[0];
    dU_s[0 * Ne_pad + jel] = v[1];
    dU_s[1 * Ne_pad + jel] = v[2];
    dU_s[2 * Ne_pad + jel] = v[3];
    d2_s[jel]            = v[4];
  }

  const double* ee_t = ee_temp + size_t(iw) * 4 * Ne_pad;
  const double* ei_t = ei_temp + size_t(iw) * 4 * Ni_pad;
  for (int a = tid; a < Nion; a += nthr)
  {
    ei_full_r[size_t(iw) * plane_ei + size_t(jel) * Ni_pad + a] = ei_t[a];
    for (int idim = 0; idim < 3; ++idim)
      ei_full_dr[size_t(iw) * 3 * plane_ei + idim * plane_ei + size_t(jel) * Ni_pad + a] =
          ei_t[(idim + 1) * Ni_pad + a];
  }
  for (int kel = tid; kel < Nelec; kel += nthr)
  {
    if (kel == jel)
      continue;
    if (kel < jel)
    {
      eer[size_t(jel) * Ne_pad + kel] = ee_t[kel];
      for (int idim = 0; idim < 3; ++idim)
        eed[idim * plane_ee + size_t(jel) * Ne_pad + kel] = ee_t[(idim + 1) * Ne_pad + kel];
    }
    else
    {
      eer[size_t(kel) * Ne_pad + jel] = ee_t[kel];
      for (int idim = 0; idim < 3; ++idim)
        eed[idim * plane_ee + size_t(kel) * Ne_pad + jel] = -ee_t[(idim + 1) * Ne_pad + kel];
    }
  }
}

__global__ void gather_grad_kernel(const int jel, const int nw, const int Ne_pad,
                                   const double* __restrict__ dUat, double* __restrict__ out)
{
  const int iw = blockIdx.x * blockDim.x + threadIdx.x;
  if (iw >= nw)
    return;
  const double* dU_s = dUat + size_t(iw) * 3 * Ne_pad;
  out[iw * 3 + 0]    = dU_s[0 * Ne_pad + jel];
  out[iw * 3 + 1]    = dU_s[1 * Ne_pad + jel];
  out[iw * 3 + 2]    = dU_s[2 * Ne_pad + jel];
}

} // namespace

DenseWorkspace::DenseWorkspace() = default;
DenseWorkspace::~DenseWorkspace() { freeAll(); }

void DenseWorkspace::ensureStream()
{
  if (!stream_)
  {
    cudaStream_t s = nullptr;
    check(cudaStreamCreate(&s), "stream create");
    stream_ = s;
  }
}

void DenseWorkspace::freeAll()
{
  if (stream_)
  {
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    stream_ = nullptr;
  }
  auto free_p = [](auto*& p) {
    if (p)
    {
      cudaFree(p);
      p = nullptr;
    }
  };
  free_p(d_ee_);
  free_p(d_ei_);
  free_p(d_fr_);
  free_p(d_fd_);
  free_p(d_ee_full_r_);
  free_p(d_ee_full_dr_);
  free_p(d_ion_cut_);
  free_p(d_gamma_);
  free_p(d_fun_cut_);
  free_p(d_vgl_);
  free_p(d_Uk_);
  free_p(d_dUk_);
  free_p(d_d2Uk_);
  free_p(d_Uat_);
  free_p(d_dUat_);
  free_p(d_d2Uat_);
  free_p(d_grad_);
  free_p(d_accept_);
  free_p(d_egrp_);
  free_p(d_igrp_);
  free_p(d_goff_);
  free_p(d_glen_);
  free_p(d_NeI_);
  free_p(d_Nee_);
  free_p(d_C_);
  cap_nw_ = cap_Ne_ = cap_Nep_ = cap_Ni_ = cap_Nip_ = cap_nfun_ = 0;
  cap_gamma_                                                    = 0;
  static_uploaded_                                              = false;
  full_resident_                                                = false;
  owner_                                                        = nullptr;
  static_ver_                                                   = 0;
}

void DenseWorkspace::ensureCapacity(int nw,
                                    int Nelec,
                                    int Ne_pad,
                                    int Nion,
                                    int Ni_pad,
                                    int nfun,
                                    size_t gamma_pool_sz)
{
  const bool need = (nw > cap_nw_) || (Nelec > cap_Ne_) || (Ne_pad > cap_Nep_) || (Nion > cap_Ni_) ||
                    (Ni_pad > cap_Nip_) || (nfun > cap_nfun_) || (gamma_pool_sz > cap_gamma_);
  if (!need)
    return;

  freeAll();

  cap_nw_    = nw;
  cap_Ne_    = Nelec;
  cap_Nep_   = Ne_pad;
  cap_Ni_    = Nion;
  cap_Nip_   = Ni_pad;
  cap_nfun_  = nfun;
  cap_gamma_ = gamma_pool_sz ? gamma_pool_sz : 1;

  const size_t ee_sz     = size_t(cap_nw_) * 4 * cap_Nep_;
  const size_t ei_sz     = size_t(cap_nw_) * 4 * cap_Nip_;
  const size_t fr_sz     = size_t(cap_nw_) * cap_Ne_ * cap_Nip_;
  const size_t fd_sz     = size_t(cap_nw_) * 3 * cap_Ne_ * cap_Nip_;
  const size_t ee_fr_sz  = size_t(cap_nw_) * cap_Ne_ * cap_Nep_;
  const size_t ee_fd_sz  = size_t(cap_nw_) * 3 * cap_Ne_ * cap_Nep_;
  const size_t vgl_sz    = size_t(cap_nw_) * 6;
  const size_t Uk_sz     = size_t(cap_nw_) * cap_Nep_;
  const size_t dUk_sz    = size_t(cap_nw_) * 3 * cap_Nep_;
  const size_t Uat_sz    = size_t(cap_nw_) * cap_Nep_;
  const size_t dUat_sz   = size_t(cap_nw_) * 3 * cap_Nep_;

  check(cudaMalloc(&d_ee_, ee_sz * sizeof(double)), "malloc ee");
  check(cudaMalloc(&d_ei_, ei_sz * sizeof(double)), "malloc ei");
  check(cudaMalloc(&d_fr_, fr_sz * sizeof(double)), "malloc fr");
  check(cudaMalloc(&d_fd_, fd_sz * sizeof(double)), "malloc fd");
  check(cudaMalloc(&d_ee_full_r_, ee_fr_sz * sizeof(double)), "malloc ee_full_r");
  check(cudaMalloc(&d_ee_full_dr_, ee_fd_sz * sizeof(double)), "malloc ee_full_dr");
  check(cudaMalloc(&d_egrp_, cap_Ne_ * sizeof(int)), "malloc egrp");
  check(cudaMalloc(&d_igrp_, cap_Ni_ * sizeof(int)), "malloc igrp");
  check(cudaMalloc(&d_ion_cut_, cap_Ni_ * sizeof(double)), "malloc ion_cut");
  check(cudaMalloc(&d_gamma_, cap_gamma_ * sizeof(double)), "malloc gamma");
  check(cudaMalloc(&d_goff_, cap_nfun_ * sizeof(int)), "malloc goff");
  check(cudaMalloc(&d_glen_, cap_nfun_ * sizeof(int)), "malloc glen");
  check(cudaMalloc(&d_fun_cut_, cap_nfun_ * sizeof(double)), "malloc fun_cut");
  check(cudaMalloc(&d_NeI_, cap_nfun_ * sizeof(int)), "malloc NeI");
  check(cudaMalloc(&d_Nee_, cap_nfun_ * sizeof(int)), "malloc Nee");
  check(cudaMalloc(&d_C_, cap_nfun_ * sizeof(int)), "malloc C");
  check(cudaMalloc(&d_vgl_, vgl_sz * sizeof(double)), "malloc vgl");
  check(cudaMalloc(&d_Uk_, Uk_sz * sizeof(double)), "malloc Uk");
  check(cudaMalloc(&d_dUk_, dUk_sz * sizeof(double)), "malloc dUk");
  check(cudaMalloc(&d_d2Uk_, Uk_sz * sizeof(double)), "malloc d2Uk");
  check(cudaMalloc(&d_Uat_, Uat_sz * sizeof(double)), "malloc Uat");
  check(cudaMalloc(&d_dUat_, dUat_sz * sizeof(double)), "malloc dUat");
  check(cudaMalloc(&d_d2Uat_, Uat_sz * sizeof(double)), "malloc d2Uat");
  check(cudaMalloc(&d_grad_, size_t(cap_nw_) * 3 * sizeof(double)), "malloc grad");
  check(cudaMalloc(&d_accept_, size_t(cap_nw_) * sizeof(int)), "malloc accept");
}

void DenseWorkspace::uploadStatic(int Nelec,
                                  int Nion,
                                  int nfun,
                                  size_t gamma_pool_sz,
                                  const int* e_grp,
                                  const int* i_grp,
                                  const double* ion_cut,
                                  const double* gamma_pool,
                                  const int* gamma_offset,
                                  const int* gamma_len,
                                  const double* fun_cut,
                                  const int* fun_NeI,
                                  const int* fun_Nee,
                                  const int* fun_C)
{
  ensureStream();
  auto s = static_cast<cudaStream_t>(stream_);
  check(cudaMemcpyAsync(d_egrp_, e_grp, Nelec * sizeof(int), cudaMemcpyHostToDevice, s), "H2D egrp");
  check(cudaMemcpyAsync(d_igrp_, i_grp, Nion * sizeof(int), cudaMemcpyHostToDevice, s), "H2D igrp");
  check(cudaMemcpyAsync(d_ion_cut_, ion_cut, Nion * sizeof(double), cudaMemcpyHostToDevice, s), "H2D ion_cut");
  if (gamma_pool_sz)
    check(cudaMemcpyAsync(d_gamma_, gamma_pool, gamma_pool_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D gamma");
  check(cudaMemcpyAsync(d_goff_, gamma_offset, nfun * sizeof(int), cudaMemcpyHostToDevice, s), "H2D goff");
  check(cudaMemcpyAsync(d_glen_, gamma_len, nfun * sizeof(int), cudaMemcpyHostToDevice, s), "H2D glen");
  check(cudaMemcpyAsync(d_fun_cut_, fun_cut, nfun * sizeof(double), cudaMemcpyHostToDevice, s), "H2D fun_cut");
  check(cudaMemcpyAsync(d_NeI_, fun_NeI, nfun * sizeof(int), cudaMemcpyHostToDevice, s), "H2D NeI");
  check(cudaMemcpyAsync(d_Nee_, fun_Nee, nfun * sizeof(int), cudaMemcpyHostToDevice, s), "H2D Nee");
  check(cudaMemcpyAsync(d_C_, fun_C, nfun * sizeof(int), cudaMemcpyHostToDevice, s), "H2D C");
  check(cudaStreamSynchronize(s), "static sync");
  static_uploaded_ = true;
}

void DenseWorkspace::uploadFull(int nw, int Nelec, int Ni_pad, const double* ei_full_r, const double* ei_full_dr)
{
  ensureStream();
  auto s               = static_cast<cudaStream_t>(stream_);
  const size_t fr_sz   = size_t(nw) * Nelec * Ni_pad;
  const size_t fd_sz   = size_t(nw) * 3 * Nelec * Ni_pad;
  check(cudaMemcpyAsync(d_fr_, ei_full_r, fr_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D fr");
  check(cudaMemcpyAsync(d_fd_, ei_full_dr, fd_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D fd");
  // leave stream open; caller syncs after launch
  full_resident_ = true;
}

void DenseWorkspace::updateEiRow(const void* owner, int iw, int jel, int Nelec, int Ni_pad, int Nion,
                                 const double* r_row, const double* dr_row_3xNi)
{
  if (!ownsFull(owner) || !d_fr_)
    return;
  ensureStream();
  auto s = static_cast<cudaStream_t>(stream_);
  // fr layout: [iw][jel][Ni_pad]
  double* dest_r = d_fr_ + (size_t(iw) * Nelec + jel) * Ni_pad;
  check(cudaMemcpyAsync(dest_r, r_row, Nion * sizeof(double), cudaMemcpyHostToDevice, s), "H2D ei row r");
  for (int idim = 0; idim < 3; ++idim)
  {
    double* dest_d = d_fd_ + ((size_t(iw) * 3 + idim) * Nelec + jel) * Ni_pad;
    check(cudaMemcpyAsync(dest_d, dr_row_3xNi + idim * Ni_pad, Nion * sizeof(double), cudaMemcpyHostToDevice, s),
          "H2D ei row dr");
  }
  // no sync: next ratioGrad on same stream sees ordered writes
}

void DenseWorkspace::launchRatioGrad(int jel,
                                     int nw,
                                     int Nelec,
                                     int Ne_pad,
                                     int Nion,
                                     int Ni_pad,
                                     int eGroups,
                                     int nfun,
                                     const double* ee_temp,
                                     const double* ei_temp,
                                     const double* ei_full_r,
                                     const double* ei_full_dr,
                                     bool upload_full,
                                     double* vgl,
                                     double* Uk,
                                     double* dUk,
                                     double* d2Uk)
{
  if (!static_uploaded_)
    throw std::runtime_error("JeeIDenseCUDA: static tables not uploaded");

  ensureStream();
  auto s = static_cast<cudaStream_t>(stream_);

  const size_t ee_sz  = size_t(nw) * 4 * Ne_pad;
  const size_t ei_sz  = size_t(nw) * 4 * Ni_pad;
  const size_t vgl_sz = size_t(nw) * 6;
  const size_t Uk_sz  = size_t(nw) * Ne_pad;
  const size_t dUk_sz = size_t(nw) * 3 * Ne_pad;

  // Pipeline: H2D temps (+ optional full) -> kernel -> D2H; single stream sync at end.
  // Host pointers should be pinned (OffloadPinned MultiWalkerMem) for async overlap.
  check(cudaMemcpyAsync(d_ee_, ee_temp, ee_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D ee");
  check(cudaMemcpyAsync(d_ei_, ei_temp, ei_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D ei");
  if (upload_full || !full_resident_)
    uploadFull(nw, Nelec, Ni_pad, ei_full_r, ei_full_dr);

  int block = 32;
  while (block < Nelec && block < 256)
    block *= 2;
  dense_ratio_grad_kernel<<<nw, block, 0, s>>>(jel, nw, Nelec, Ne_pad, Nion, Ni_pad, eGroups, nfun, d_ee_, d_ei_, d_fr_,
                                               d_fd_, d_egrp_, d_igrp_, d_ion_cut_, d_gamma_, d_goff_, d_fun_cut_, d_NeI_,
                                               d_Nee_, d_C_, d_Uat_, d_vgl_, d_Uk_, d_dUk_, d_d2Uk_);
  check(cudaGetLastError(), "ratioGrad kernel");

  check(cudaMemcpyAsync(vgl, d_vgl_, vgl_sz * sizeof(double), cudaMemcpyDeviceToHost, s), "D2H vgl");
  check(cudaMemcpyAsync(Uk, d_Uk_, Uk_sz * sizeof(double), cudaMemcpyDeviceToHost, s), "D2H Uk");
  check(cudaMemcpyAsync(dUk, d_dUk_, dUk_sz * sizeof(double), cudaMemcpyDeviceToHost, s), "D2H dUk");
  check(cudaMemcpyAsync(d2Uk, d_d2Uk_, Uk_sz * sizeof(double), cudaMemcpyDeviceToHost, s), "D2H d2Uk");
  check(cudaStreamSynchronize(s), "ratioGrad stream sync");
}

void DenseWorkspace::launchRecompute(int nw,
                                     int Nelec,
                                     int Ne_pad,
                                     int Nion,
                                     int Ni_pad,
                                     int eGroups,
                                     int nfun,
                                     const double* ee_full_r,
                                     const double* ee_full_dr,
                                     const double* ei_full_r,
                                     const double* ei_full_dr,
                                     double* Uat,
                                     double* dUat,
                                     double* d2Uat)
{
  if (!static_uploaded_)
    throw std::runtime_error("JeeIDenseCUDA: static tables not uploaded");

  ensureStream();
  auto s = static_cast<cudaStream_t>(stream_);

  const size_t ee_fr_sz = size_t(nw) * Nelec * Ne_pad;
  const size_t ee_fd_sz = size_t(nw) * 3 * Nelec * Ne_pad;
  const size_t fr_sz    = size_t(nw) * Nelec * Ni_pad;
  const size_t fd_sz    = size_t(nw) * 3 * Nelec * Ni_pad;
  const size_t U_sz     = size_t(nw) * Ne_pad;
  const size_t dU_sz    = size_t(nw) * 3 * Ne_pad;

  check(cudaMemcpyAsync(d_ee_full_r_, ee_full_r, ee_fr_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D ee_full_r");
  check(cudaMemcpyAsync(d_ee_full_dr_, ee_full_dr, ee_fd_sz * sizeof(double), cudaMemcpyHostToDevice, s),
        "H2D ee_full_dr");
  check(cudaMemcpyAsync(d_fr_, ei_full_r, fr_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D ei_full_r recompute");
  check(cudaMemcpyAsync(d_fd_, ei_full_dr, fd_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D ei_full_dr recompute");
  full_resident_ = true;

  int block = 32;
  while (block < Nelec && block < 256)
    block *= 2;
  dense_recompute_kernel<<<nw, block, 0, s>>>(nw, Nelec, Ne_pad, Nion, Ni_pad, eGroups, nfun, d_ee_full_r_,
                                              d_ee_full_dr_, d_fr_, d_fd_, d_egrp_, d_igrp_, d_ion_cut_, d_gamma_,
                                              d_goff_, d_fun_cut_, d_NeI_, d_Nee_, d_C_, d_Uat_, d_dUat_, d_d2Uat_);
  check(cudaGetLastError(), "recompute kernel");

  check(cudaMemcpyAsync(Uat, d_Uat_, U_sz * sizeof(double), cudaMemcpyDeviceToHost, s), "D2H Uat");
  check(cudaMemcpyAsync(dUat, d_dUat_, dU_sz * sizeof(double), cudaMemcpyDeviceToHost, s), "D2H dUat");
  check(cudaMemcpyAsync(d2Uat, d_d2Uat_, U_sz * sizeof(double), cudaMemcpyDeviceToHost, s), "D2H d2Uat");
  check(cudaStreamSynchronize(s), "recompute stream sync");
}


void DenseWorkspace::uploadFullGeometry(int nw, int Nelec, int Ne_pad, int Ni_pad, const double* ee_full_r,
                                        const double* ee_full_dr, const double* ei_full_r, const double* ei_full_dr,
                                        const double* uat_state)
{
  ensureStream();
  auto s                = static_cast<cudaStream_t>(stream_);
  const size_t ee_fr_sz = size_t(nw) * Nelec * Ne_pad;
  const size_t ee_fd_sz = size_t(nw) * 3 * Nelec * Ne_pad;
  const size_t fr_sz    = size_t(nw) * Nelec * Ni_pad;
  const size_t fd_sz    = size_t(nw) * 3 * Nelec * Ni_pad;
  const size_t U_sz     = size_t(nw) * Ne_pad;
  check(cudaMemcpyAsync(d_ee_full_r_, ee_full_r, ee_fr_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D geo ee r");
  check(cudaMemcpyAsync(d_ee_full_dr_, ee_full_dr, ee_fd_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D geo ee d");
  check(cudaMemcpyAsync(d_fr_, ei_full_r, fr_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D geo ei r");
  check(cudaMemcpyAsync(d_fd_, ei_full_dr, fd_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D geo ei d");
  // uat_state uses the pooled mw_allUat layout: [Uat | dUat | d2Uat] batches back-to-back
  check(cudaMemcpyAsync(d_Uat_, uat_state, U_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D state U");
  check(cudaMemcpyAsync(d_dUat_, uat_state + U_sz, 3 * U_sz * sizeof(double), cudaMemcpyHostToDevice, s), "H2D state dU");
  check(cudaMemcpyAsync(d_d2Uat_, uat_state + 4 * U_sz, U_sz * sizeof(double), cudaMemcpyHostToDevice, s),
        "H2D state d2U");
  full_resident_ = true;
}

void DenseWorkspace::downloadState(int nw, int Ne_pad, double* uat_state)
{
  ensureStream();
  auto s            = static_cast<cudaStream_t>(stream_);
  const size_t U_sz = size_t(nw) * Ne_pad;
  check(cudaMemcpyAsync(uat_state, d_Uat_, U_sz * sizeof(double), cudaMemcpyDeviceToHost, s), "D2H state U");
  check(cudaMemcpyAsync(uat_state + U_sz, d_dUat_, 3 * U_sz * sizeof(double), cudaMemcpyDeviceToHost, s), "D2H state dU");
  check(cudaMemcpyAsync(uat_state + 4 * U_sz, d_d2Uat_, U_sz * sizeof(double), cudaMemcpyDeviceToHost, s),
        "D2H state d2U");
  check(cudaStreamSynchronize(s), "state sync");
}

void DenseWorkspace::launchAccept(const void* owner, int jel, int nw, int Nelec, int Ne_pad, int Nion, int Ni_pad,
                                  int eGroups, int nfun, const int* accepted)
{
  if (!ownsFull(owner))
    throw std::runtime_error("JeeIDenseCUDA: accept without resident tables");
  ensureStream();
  auto s = static_cast<cudaStream_t>(stream_);
  check(cudaMemcpyAsync(d_accept_, accepted, nw * sizeof(int), cudaMemcpyHostToDevice, s), "H2D accept flags");
  int block = 32;
  while (block < Nelec && block < 256)
    block *= 2;
  dense_accept_kernel<<<nw, block, 0, s>>>(jel, nw, Nelec, Ne_pad, Nion, Ni_pad, eGroups, nfun, d_accept_, d_ee_, d_ei_,
                                           d_ee_full_r_, d_ee_full_dr_, d_fr_, d_fd_, d_egrp_, d_igrp_, d_ion_cut_,
                                           d_gamma_, d_goff_, d_fun_cut_, d_NeI_, d_Nee_, d_C_, d_vgl_, d_Uk_, d_dUk_,
                                           d_d2Uk_, d_Uat_, d_dUat_, d_d2Uat_);
  check(cudaGetLastError(), "accept kernel");
  // no sync: subsequent same-stream work sees ordered state
}

void DenseWorkspace::gatherGrad(const void* owner, int jel, int nw, int Ne_pad, double* grad3)
{
  if (!ownsFull(owner))
    throw std::runtime_error("JeeIDenseCUDA: gatherGrad without resident state");
  ensureStream();
  auto s = static_cast<cudaStream_t>(stream_);
  gather_grad_kernel<<<(nw + 127) / 128, 128, 0, s>>>(jel, nw, Ne_pad, d_dUat_, d_grad_);
  check(cudaGetLastError(), "grad kernel");
  check(cudaMemcpyAsync(grad3, d_grad_, size_t(nw) * 3 * sizeof(double), cudaMemcpyDeviceToHost, s), "D2H grad");
  check(cudaStreamSynchronize(s), "grad sync");
}

DenseWorkspace& default_workspace()
{
  // One workspace per host thread: batched drivers run crowds in parallel OpenMP.
  // Sharing a single process-wide workspace races cudaMalloc/Memcpy across crowds.
  thread_local DenseWorkspace ws;
  return ws;
}

static void ensure_static(DenseWorkspace& ws,
                          const void* owner,
                          unsigned long long static_ver,
                          int nw,
                          int Nelec,
                          int Ne_pad,
                          int Nion,
                          int Ni_pad,
                          int nfun,
                          const int* e_grp,
                          const int* i_grp,
                          const double* ion_cut,
                          const double* gamma_pool,
                          const int* gamma_offset,
                          const int* gamma_len,
                          const double* fun_cut,
                          const int* fun_NeI,
                          const int* fun_Nee,
                          const int* fun_C)
{
  size_t gamma_pool_sz = 0;
  for (int f = 0; f < nfun; ++f)
    if (gamma_len[f] > 0)
      gamma_pool_sz += size_t(gamma_len[f]);
  ws.ensureCapacity(nw, Nelec, Ne_pad, Nion, Ni_pad, nfun, gamma_pool_sz);
  if (!ws.matchesStatic(owner, static_ver))
  {
    // Different owner: any resident full tables belong to someone else.
    if (!ws.sameOwner(owner))
      ws.invalidateFull();
    ws.uploadStatic(Nelec, Nion, nfun, gamma_pool_sz, e_grp, i_grp, ion_cut, gamma_pool, gamma_offset, gamma_len,
                    fun_cut, fun_NeI, fun_Nee, fun_C);
    ws.setOwner(owner, static_ver);
  }
}

DenseWorkspace& prepare_dense_workspace(const void* owner,
                                        unsigned long long static_ver,
                                        int nw,
                                        int Nelec,
                                        int Ne_pad,
                                        int Nion,
                                        int Ni_pad,
                                        int nfun,
                                        const int* e_grp,
                                        const int* i_grp,
                                        const double* ion_cut,
                                        const double* gamma_pool,
                                        const int* gamma_offset,
                                        const int* gamma_len,
                                        const double* fun_cut,
                                        const int* fun_NeI,
                                        const int* fun_Nee,
                                        const int* fun_C)
{
  auto& ws = default_workspace();
  ensure_static(ws, owner, static_ver, nw, Nelec, Ne_pad, Nion, Ni_pad, nfun, e_grp, i_grp, ion_cut, gamma_pool,
                gamma_offset, gamma_len, fun_cut, fun_NeI, fun_Nee, fun_C);
  return ws;
}

void launch_dense_ratio_grad(const void* owner,
                             unsigned long long static_ver,
                             int jel,
                             int nw,
                             int Nelec,
                             int Ne_pad,
                             int Nion,
                             int Ni_pad,
                             int eGroups,
                             int /*iGroups*/,
                             int nfun,
                             const double* ee_temp,
                             const double* ei_temp,
                             const double* ei_full_r,
                             const double* ei_full_dr,
                             const int* e_grp,
                             const int* i_grp,
                             const double* ion_cut,
                             const double* gamma_pool,
                             const int* gamma_offset,
                             const int* gamma_len,
                             const double* fun_cut,
                             const int* fun_NeI,
                             const int* fun_Nee,
                             const int* fun_C,
                             bool force_full_upload,
                             double* vgl,
                             double* Uk,
                             double* dUk,
                             double* d2Uk)
{
  auto& ws = default_workspace();
  ensure_static(ws, owner, static_ver, nw, Nelec, Ne_pad, Nion, Ni_pad, nfun, e_grp, i_grp, ion_cut, gamma_pool,
                gamma_offset, gamma_len, fun_cut, fun_NeI, fun_Nee, fun_C);
  const bool need_full = force_full_upload || !ws.hasFull();
  ws.launchRatioGrad(jel, nw, Nelec, Ne_pad, Nion, Ni_pad, eGroups, nfun, ee_temp, ei_temp, ei_full_r, ei_full_dr,
                     need_full, vgl, Uk, dUk, d2Uk);
}

void launch_dense_recompute(const void* owner,
                            unsigned long long static_ver,
                            int nw,
                            int Nelec,
                            int Ne_pad,
                            int Nion,
                            int Ni_pad,
                            int eGroups,
                            int /*iGroups*/,
                            int nfun,
                            const double* ee_full_r,
                            const double* ee_full_dr,
                            const double* ei_full_r,
                            const double* ei_full_dr,
                            const int* e_grp,
                            const int* i_grp,
                            const double* ion_cut,
                            const double* gamma_pool,
                            const int* gamma_offset,
                            const int* gamma_len,
                            const double* fun_cut,
                            const int* fun_NeI,
                            const int* fun_Nee,
                            const int* fun_C,
                            double* Uat,
                            double* dUat,
                            double* d2Uat)
{
  auto& ws = default_workspace();
  ensure_static(ws, owner, static_ver, nw, Nelec, Ne_pad, Nion, Ni_pad, nfun, e_grp, i_grp, ion_cut, gamma_pool,
                gamma_offset, gamma_len, fun_cut, fun_NeI, fun_Nee, fun_C);
  ws.launchRecompute(nw, Nelec, Ne_pad, Nion, Ni_pad, eGroups, nfun, ee_full_r, ee_full_dr, ei_full_r, ei_full_dr, Uat,
                     dUat, d2Uat);
}

} // namespace jeei_cuda
} // namespace qmcplusplus
