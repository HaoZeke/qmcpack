//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: QMCPACK developers
//
// File created by: QMCPACK developers
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-

#include "JeeIOrbitalSoA.h"
#include "ResourceCollection.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include "PolynomialFunctor3DOffload.h"
#include "config.h"
#include <algorithm>
#if defined(ENABLE_CUDA)
#include <atomic>
#include "JeeIDenseCUDA.h"
#endif

namespace qmcplusplus
{

template<typename T>
struct JeeIOrbitalSoAMultiWalkerMem : public Resource
{
  Vector<char, OffloadPinnedAllocator<char>> mw_update_buffer;
  Vector<char, OffloadPinnedAllocator<char>> mw_ratiograd_buffer;
  Vector<char, OffloadPinnedAllocator<char>> transfer_buffer;
  Vector<T, OffloadPinnedAllocator<T>> mw_vals;
  Matrix<T, OffloadPinnedAllocator<T>> mw_vgl;
  Vector<T, OffloadPinnedAllocator<T>> mw_allUat;
  Vector<T, OffloadPinnedAllocator<T>> mw_cur_allu;
  // packed dual-table distances for mw_ratioGrad
  Vector<T, OffloadPinnedAllocator<T>> mw_ee_temp;     // [nw][1+DIM][Ne_pad]
  Vector<T, OffloadPinnedAllocator<T>> mw_ei_temp;     // [nw][1+DIM][Ni_pad]
  Vector<T, OffloadPinnedAllocator<T>> mw_ei_full_r;   // [nw][Nelec][Ni_pad]
  Vector<T, OffloadPinnedAllocator<T>> mw_ei_full_dr;  // [nw][DIM][Nelec][Ni_pad]
  Vector<T, OffloadPinnedAllocator<T>> mw_ee_full_r;   // [nw][Nelec][Ne_pad] AA lower triangle
  Vector<T, OffloadPinnedAllocator<T>> mw_ee_full_dr;  // [nw][DIM][Nelec][Ne_pad]
  Vector<T, OffloadPinnedAllocator<T>> mw_Uk;          // [nw][Ne_pad]
  Vector<T, OffloadPinnedAllocator<T>> mw_dUk;         // [nw][DIM][Ne_pad]
  Vector<T, OffloadPinnedAllocator<T>> mw_d2Uk;        // [nw][Ne_pad]
  Vector<T, OffloadPinnedAllocator<T>> mw_Uat_batch;   // [nw][Ne_pad] recompute output
  Vector<T, OffloadPinnedAllocator<T>> mw_dUat_batch;  // [nw][DIM][Ne_pad]
  Vector<T, OffloadPinnedAllocator<T>> mw_d2Uat_batch; // [nw][Ne_pad]
  Vector<int, OffloadPinnedAllocator<int>> e_grp;
  Vector<int, OffloadPinnedAllocator<int>> i_grp;
  Vector<T, OffloadPinnedAllocator<T>> ion_cutoff;
  // CUDA static pack (species / gamma / cutoffs); version bumps whenever the pack
  // content changes so the device workspace can detect stale tables.
  bool cuda_static_ready              = false;
  unsigned long long cuda_static_version = 0;
  // Full e-I tables dirty after recompute/accept; ratioGrad can skip pack+H2D when clean
  bool cuda_full_dirty = true;
  std::vector<int> cuda_gamma_offset;
  std::vector<int> cuda_gamma_len;
  std::vector<double> cuda_fun_cut;
  std::vector<int> cuda_fun_NeI;
  std::vector<int> cuda_fun_Nee;
  std::vector<int> cuda_fun_C;
  std::vector<double> cuda_gamma_pool;

  JeeIOrbitalSoAMultiWalkerMem() : Resource("JeeIOrbitalSoAMultiWalkerMem") {}
  JeeIOrbitalSoAMultiWalkerMem(const JeeIOrbitalSoAMultiWalkerMem&) : JeeIOrbitalSoAMultiWalkerMem() {}
  std::unique_ptr<Resource> makeClone() const override
  {
    return std::make_unique<JeeIOrbitalSoAMultiWalkerMem>(*this);
  }
};

#if defined(ENABLE_CUDA)
namespace
{
/// process-wide pack version source: unique per rebuild, so an old owner freed and a
/// new one allocated at the same address can never alias a stale device static pack
std::atomic<unsigned long long> jeei_cuda_pack_counter{1};

/** Rebuild the flat gamma/cutoff/species pack for the CUDA kernels.
 *  With force=false only the first build runs (hot path); with force=true the pack is
 *  rebuilt and the version bumps when content changed (parameter updates between blocks).
 */
template<typename FT, typename T>
void refresh_cuda_static_pack(const Array<FT*, 3>& F,
                              int iG,
                              int eG,
                              JeeIOrbitalSoAMultiWalkerMem<T>& mem,
                              bool force)
{
  if (mem.cuda_static_ready && !force)
    return;
  const int nfun = iG * eG * eG;
  std::vector<int> goff(nfun, -1), glen(nfun, 0), gNeI(nfun, 0), gNee(nfun, 0), gC(nfun, 0);
  std::vector<double> gcut(nfun, 0.0), gpool;
  for (int ig = 0; ig < iG; ++ig)
    for (int jg = 0; jg < eG; ++jg)
      for (int kg = 0; kg < eG; ++kg)
      {
        const int fid = (ig * eG + jg) * eG + kg;
        FT* f         = F(ig, jg, kg);
        if (!f || f->gammaOffloadSize() == 0)
          continue;
        goff[fid]      = static_cast<int>(gpool.size());
        glen[fid]      = f->gammaOffloadSize();
        const auto* gp = f->gammaOffloadData();
        gpool.insert(gpool.end(), gp, gp + glen[fid]);
        gcut[fid] = static_cast<double>(f->cutoff_radius);
        gNeI[fid] = f->N_eI;
        gNee[fid] = f->N_ee;
        gC[fid]   = f->C;
      }
  const bool changed = !mem.cuda_static_ready || gpool != mem.cuda_gamma_pool || goff != mem.cuda_gamma_offset ||
      glen != mem.cuda_gamma_len || gcut != mem.cuda_fun_cut || gNeI != mem.cuda_fun_NeI ||
      gNee != mem.cuda_fun_Nee || gC != mem.cuda_fun_C;
  if (!changed)
    return;
  mem.cuda_gamma_offset   = std::move(goff);
  mem.cuda_gamma_len      = std::move(glen);
  mem.cuda_fun_cut        = std::move(gcut);
  mem.cuda_fun_NeI        = std::move(gNeI);
  mem.cuda_fun_Nee        = std::move(gNee);
  mem.cuda_fun_C          = std::move(gC);
  mem.cuda_gamma_pool     = std::move(gpool);
  mem.cuda_static_version = jeei_cuda_pack_counter.fetch_add(1);
  mem.cuda_static_ready   = true;
}
} // namespace
#endif

template<typename FT>
JeeIOrbitalSoA<FT>::JeeIOrbitalSoA(const std::string& obj_name,
                                  const ParticleSet& ions,
                                  ParticleSet& elecs,
                                  bool use_offload)
    : WaveFunctionComponent(obj_name),
      use_offload_(use_offload),
      // Host-visible ee/eI tables: dense dual-table offload packs distances on the host
      // into OffloadPinned buffers, then evaluates under PRAGMA_OFFLOAD. DTModes::ALL_OFF
      // (device-resident multi-walker DTs) is the next stage once AA/AB temp+full device
      // pointers are consumed end-to-end like TwoBodyJastrow.
      ee_Table_ID_(elecs.addTable(elecs, DTModes::NEED_TEMP_DATA_ON_HOST | DTModes::NEED_VP_FULL_TABLE_ON_HOST)),
      ei_Table_ID_(elecs.addTable(ions, DTModes::NEED_FULL_TABLE_ANYTIME | DTModes::NEED_VP_FULL_TABLE_ON_HOST)),
      Ions(ions)
{
  if (my_name_.empty())
    throw std::runtime_error("JeeIOrbitalSoA object name cannot be empty!");
  // Offload flag means multi-walker accelerated path (OMPTarget and/or CUDA dense).
  // CUDA path only needs gamma_offload_ tables (PolynomialFunctor3D always has them).
  if (use_offload_ && !FT::isOMPoffload())
  {
#if !defined(ENABLE_CUDA)
    throw std::runtime_error("JeeIOrbitalSoA offload requested but functor does not support OpenMP offload");
#endif
  }
  init(elecs);
}

template<typename FT>
JeeIOrbitalSoA<FT>::~JeeIOrbitalSoA() = default;

template<typename FT>
void JeeIOrbitalSoA<FT>::createResource(ResourceCollection& collection) const
{
  collection.addResource(std::make_unique<JeeIOrbitalSoAMultiWalkerMem<RealType>>());
}

template<typename FT>
void JeeIOrbitalSoA<FT>::acquireResource(ResourceCollection& collection,
                                         const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  auto& wfc_leader          = wfc_list.getCastedLeader<JeeIOrbitalSoA<FT>>();
  wfc_leader.mw_mem_handle_ = collection.lendResource<JeeIOrbitalSoAMultiWalkerMem<RealType>>();
  const size_t nw           = wfc_list.size();
  const size_t N_padded     = wfc_leader.Nelec_padded;
  const size_t N            = wfc_leader.Nelec;
  auto& mw_allUat           = wfc_leader.mw_mem_handle_.getResource().mw_allUat;
  mw_allUat.resize(N_padded * (OHMMS_DIM + 2) * nw);
  for (size_t iw = 0; iw < nw; iw++)
  {
    auto& wfc = wfc_list.getCastedElement<JeeIOrbitalSoA<FT>>(iw);

    Vector<valT> Uat_view(mw_allUat.data() + iw * N_padded, N);
    Uat_view = wfc.Uat;
    wfc.Uat.free();
    wfc.Uat.attachReference(mw_allUat.data() + iw * N_padded, N);

    gContainer_type dUat_view(mw_allUat.data() + nw * N_padded + iw * N_padded * OHMMS_DIM, N, N_padded);
    dUat_view = wfc.dUat;
    wfc.dUat.free();
    wfc.dUat.attachReference(N, N_padded, mw_allUat.data() + nw * N_padded + iw * N_padded * OHMMS_DIM);

    Vector<valT> d2Uat_view(mw_allUat.data() + nw * N_padded * (OHMMS_DIM + 1) + iw * N_padded, N);
    d2Uat_view = wfc.d2Uat;
    wfc.d2Uat.free();
    wfc.d2Uat.attachReference(mw_allUat.data() + nw * N_padded * (OHMMS_DIM + 1) + iw * N_padded, N);
  }
  wfc_leader.mw_mem_handle_.getResource().mw_cur_allu.resize(N_padded * 3 * nw);
}

template<typename FT>
void JeeIOrbitalSoA<FT>::releaseResource(ResourceCollection& collection,
                                         const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  auto& wfc_leader = wfc_list.getCastedLeader<JeeIOrbitalSoA<FT>>();
  const size_t nw  = wfc_list.size();
  const size_t N_padded = wfc_leader.Nelec_padded;
  const size_t N        = wfc_leader.Nelec;
  auto& mw_allUat  = wfc_leader.mw_mem_handle_.getResource().mw_allUat;
  for (size_t iw = 0; iw < nw; iw++)
  {
    auto& wfc = wfc_list.getCastedElement<JeeIOrbitalSoA<FT>>(iw);

    Vector<valT> Uat_view(mw_allUat.data() + iw * N_padded, N);
    wfc.Uat.free();
    wfc.Uat.resize(N);
    wfc.Uat = Uat_view;

    gContainer_type dUat_view(mw_allUat.data() + nw * N_padded + iw * N_padded * OHMMS_DIM, N, N_padded);
    wfc.dUat.free();
    wfc.dUat.resize(N);
    wfc.dUat = dUat_view;

    Vector<valT> d2Uat_view(mw_allUat.data() + nw * N_padded * (OHMMS_DIM + 1) + iw * N_padded, N);
    wfc.d2Uat.free();
    wfc.d2Uat.resize(N);
    wfc.d2Uat = d2Uat_view;
  }
  collection.takebackResource(wfc_leader.mw_mem_handle_);
}

template<typename FT>
void JeeIOrbitalSoA<FT>::mw_evaluateLog(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                        const RefVectorWithLeader<ParticleSet>& p_list,
                                        const RefVector<ParticleSet::ParticleGradient>& G_list,
                                        const RefVector<ParticleSet::ParticleLaplacian>& L_list) const
{
  assert(this == &wfc_list.getLeader());
  const int nw = wfc_list.size();
  const std::vector<bool> recompute_all(nw, true);
  mw_recompute(wfc_list, p_list, recompute_all);

  for (int iw = 0; iw < nw; iw++)
  {
    auto& wfc      = wfc_list.getCastedElement<JeeIOrbitalSoA<FT>>(iw);
    wfc.log_value_ = wfc.computeGL(G_list[iw], L_list[iw]);
  }
}

template<typename FT>
void JeeIOrbitalSoA<FT>::mw_recompute(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                      const RefVectorWithLeader<ParticleSet>& p_list,
                                      const std::vector<bool>& recompute) const
{
  assert(this == &wfc_list.getLeader());
  const int nw = wfc_list.size();

#if defined(ENABLE_CUDA)
  // Cut2: CUDA dense multi-walker recompute (all walkers that need full rebuild)
  if (use_offload_)
  {
    auto& leader  = wfc_list.getCastedLeader<JeeIOrbitalSoA<FT>>();
    const int Ne  = leader.Nelec;
    const int Nep = static_cast<int>(leader.Nelec_padded);
    const int Ni  = leader.Nion;
    const int Nip = getAlignedSize<valT>(Ni);
    const int eG  = leader.eGroups;
    const int iG  = leader.iGroups;
    const int nfun = iG * eG * eG;

    bool any = false;
    for (int iw = 0; iw < nw; ++iw)
      if (recompute[iw])
        any = true;

    if (any)
    {
      JeeIOrbitalSoAMultiWalkerMem<RealType> local_mem;
      auto* mem_ptr = &local_mem;
      if (leader.mw_mem_handle_)
        mem_ptr = &leader.mw_mem_handle_.getResource();
      auto& mem = *mem_ptr;

      mem.mw_ei_full_r.resize(static_cast<size_t>(nw) * Ne * Nip);
      mem.mw_ei_full_dr.resize(static_cast<size_t>(nw) * OHMMS_DIM * Ne * Nip);
      mem.mw_ee_full_r.resize(static_cast<size_t>(nw) * Ne * Nep);
      mem.mw_ee_full_dr.resize(static_cast<size_t>(nw) * OHMMS_DIM * Ne * Nep);
      mem.mw_Uat_batch.resize(static_cast<size_t>(nw) * Nep);
      mem.mw_dUat_batch.resize(static_cast<size_t>(nw) * OHMMS_DIM * Nep);
      mem.mw_d2Uat_batch.resize(static_cast<size_t>(nw) * Nep);
      mem.e_grp.resize(Ne);
      mem.i_grp.resize(Ni);
      mem.ion_cutoff.resize(Ni);

      auto& P0 = p_list[0];
      for (int e = 0; e < Ne; ++e)
        mem.e_grp[e] = P0.GroupID[e];
      for (int i = 0; i < Ni; ++i)
      {
        mem.i_grp[i]      = leader.Ions.GroupID[i];
        mem.ion_cutoff[i] = leader.Ion_cutoff[i];
      }

      // Rebuild the static pack off the hot path: parameters may have changed since
      // the last block (optimizer updates); the version bump forces a device re-upload.
      refresh_cuda_static_pack(leader.F, iG, eG, mem, /*force=*/true);

      // Zero ee batch buffers (only the lower triangle gets packed)
      std::fill(mem.mw_ee_full_r.begin(), mem.mw_ee_full_r.end(), valT(0));
      std::fill(mem.mw_ee_full_dr.begin(), mem.mw_ee_full_dr.end(), valT(0));

      // Pack ALL walkers, not only those flagged for recompute: the launch uploads the
      // whole buffer and marks the device e-I tables resident for every walker, so a
      // skipped walker's rows must reflect its current host distance tables (previous
      // accepts were patched on device only, never into this host pack buffer).
#pragma omp parallel for schedule(static)
      for (int iw = 0; iw < nw; ++iw)
      {
        auto& P                = p_list[iw];
        const auto& ee_table   = P.getDistTableAA(leader.ee_Table_ID_);
        const auto& eI_table   = P.getDistTableAB(leader.ei_Table_ID_);
        const auto& ee_full_r  = ee_table.getDistances();
        const auto& ee_full_dr = ee_table.getDisplacements();
        const auto& ei_full_r  = eI_table.getDistances();
        const auto& ei_full_dr = eI_table.getDisplacements();

        valT* ei_r  = mem.mw_ei_full_r.data() + static_cast<size_t>(iw) * Ne * Nip;
        valT* ei_dr = mem.mw_ei_full_dr.data() + static_cast<size_t>(iw) * OHMMS_DIM * Ne * Nip;
        for (int kel = 0; kel < Ne; ++kel)
          for (int a = 0; a < Ni; ++a)
          {
            ei_r[kel * Nip + a] = ei_full_r[kel][a];
            for (int idim = 0; idim < OHMMS_DIM; ++idim)
              ei_dr[idim * Ne * Nip + kel * Nip + a] = ei_full_dr[kel].data(idim)[a];
          }

        // AA lower triangle only (getDistRow(jel) length = jel)
        valT* ee_r  = mem.mw_ee_full_r.data() + static_cast<size_t>(iw) * Ne * Nep;
        valT* ee_dr = mem.mw_ee_full_dr.data() + static_cast<size_t>(iw) * OHMMS_DIM * Ne * Nep;
        for (int jel = 0; jel < Ne; ++jel)
          for (int kel = 0; kel < jel; ++kel)
          {
            ee_r[jel * Nep + kel] = ee_full_r[jel][kel];
            for (int idim = 0; idim < OHMMS_DIM; ++idim)
              ee_dr[idim * Ne * Nep + jel * Nep + kel] = ee_full_dr[jel].data(idim)[kel];
          }
      }

      jeei_cuda::launch_dense_recompute(&mem, mem.cuda_static_version, nw, Ne, Nep, Ni, Nip, eG, iG, nfun,
                                        mem.mw_ee_full_r.data(),
                                        mem.mw_ee_full_dr.data(), mem.mw_ei_full_r.data(), mem.mw_ei_full_dr.data(),
                                        mem.e_grp.data(), mem.i_grp.data(), mem.ion_cutoff.data(),
                                        mem.cuda_gamma_pool.data(), mem.cuda_gamma_offset.data(),
                                        mem.cuda_gamma_len.data(), mem.cuda_fun_cut.data(), mem.cuda_fun_NeI.data(),
                                        mem.cuda_fun_Nee.data(), mem.cuda_fun_C.data(), mem.mw_Uat_batch.data(),
                                        mem.mw_dUat_batch.data(), mem.mw_d2Uat_batch.data());

#pragma omp parallel for schedule(static)
      for (int iw = 0; iw < nw; ++iw)
      {
        auto& wfc = wfc_list.getCastedElement<JeeIOrbitalSoA<FT>>(iw);
        if (recompute[iw])
        {
          for (int e = 0; e < Ne; ++e)
          {
            wfc.Uat[e]   = mem.mw_Uat_batch[static_cast<size_t>(iw) * Nep + e];
            wfc.d2Uat[e] = mem.mw_d2Uat_batch[static_cast<size_t>(iw) * Nep + e];
            for (int idim = 0; idim < OHMMS_DIM; ++idim)
              wfc.dUat.data(idim)[e] =
                  mem.mw_dUat_batch[static_cast<size_t>(iw) * OHMMS_DIM * Nep + idim * Nep + e];
          }
        }
        // Compact lists needed for host acceptMove
        wfc.build_compact_list(p_list[iw]);
      }

      mem.cuda_full_dirty = false; // device eI full just uploaded
      if (leader.mw_mem_handle_)
        mem.mw_allUat.updateTo();
      return;
    }
  }
#endif

  // Host multi-walker: OpenMP over walkers (dense recompute when use_offload_)
#pragma omp parallel for schedule(dynamic)
  for (int iw = 0; iw < nw; iw++)
  {
    auto& jeei = wfc_list.getCastedElement<JeeIOrbitalSoA<FT>>(iw);
    if (recompute[iw])
      jeei.recompute(p_list[iw]);
    else
      jeei.build_compact_list(p_list[iw]);
  }

  if (use_offload_)
  {
    auto& wfc_leader = wfc_list.getCastedLeader<JeeIOrbitalSoA<FT>>();
    if (wfc_leader.mw_mem_handle_)
    {
      auto& mem           = wfc_leader.mw_mem_handle_.getResource();
      mem.mw_allUat.updateTo();
      mem.cuda_full_dirty = true;
    }
#if defined(ENABLE_CUDA)
    jeei_cuda::default_workspace().invalidateFull();
#endif
  }
}

template<typename FT>
void JeeIOrbitalSoA<FT>::mw_ratioGrad_offload(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                              const RefVectorWithLeader<ParticleSet>& p_list,
                                              int iat,
                                              std::vector<PsiValue>& ratios,
                                              std::vector<GradType>* grad_new,
                                              bool need_grad) const
{
  auto& leader  = wfc_list.getCastedLeader<JeeIOrbitalSoA<FT>>();
  const int nw  = wfc_list.size();
  const int Ne  = leader.Nelec;
  const int Nep = static_cast<int>(leader.Nelec_padded);
  const int Ni  = leader.Nion;
  const int Nip = getAlignedSize<valT>(Ni);
  const int eG  = leader.eGroups;
  const int iG  = leader.iGroups;
  const int nfun = iG * eG * eG;

  JeeIOrbitalSoAMultiWalkerMem<RealType> local_mem;
  JeeIOrbitalSoAMultiWalkerMem<RealType>* mem_ptr = &local_mem;
  if (leader.mw_mem_handle_)
    mem_ptr = &leader.mw_mem_handle_.getResource();
  auto& mem = *mem_ptr;

  mem.mw_vgl.resize(nw, OHMMS_DIM + 2);
  mem.mw_ee_temp.resize(static_cast<size_t>(nw) * (OHMMS_DIM + 1) * Nep);
  mem.mw_ei_temp.resize(static_cast<size_t>(nw) * (OHMMS_DIM + 1) * Nip);
  mem.mw_ei_full_r.resize(static_cast<size_t>(nw) * Ne * Nip);
  mem.mw_ei_full_dr.resize(static_cast<size_t>(nw) * OHMMS_DIM * Ne * Nip);
  mem.mw_Uk.resize(static_cast<size_t>(nw) * Nep);
  mem.mw_dUk.resize(static_cast<size_t>(nw) * OHMMS_DIM * Nep);
  mem.mw_d2Uk.resize(static_cast<size_t>(nw) * Nep);
  mem.e_grp.resize(Ne);
  mem.i_grp.resize(Ni);
  mem.ion_cutoff.resize(Ni);

  auto& P0 = p_list[0];
  for (int e = 0; e < Ne; ++e)
    mem.e_grp[e] = P0.GroupID[e];
  for (int i = 0; i < Ni; ++i)
  {
    mem.i_grp[i]      = leader.Ions.GroupID[i];
    mem.ion_cutoff[i] = leader.Ion_cutoff[i];
  }
  mem.e_grp.updateTo();
  mem.i_grp.updateTo();
  mem.ion_cutoff.updateTo();

  std::vector<const valT*> gamma_host(nfun, nullptr);
  std::vector<valT> fun_cut(nfun, valT(0));
  std::vector<int> fun_NeI(nfun, 0), fun_Nee(nfun, 0), fun_C(nfun, 0);
  for (int ig = 0; ig < iG; ++ig)
    for (int jg = 0; jg < eG; ++jg)
      for (int kg = 0; kg < eG; ++kg)
      {
        const int fid = (ig * eG + jg) * eG + kg;
        FT* f         = leader.F(ig, jg, kg);
        if (!f)
          continue;
        gamma_host[fid] = f->gammaOffloadData();
        fun_cut[fid]    = static_cast<valT>(f->cutoff_radius);
        fun_NeI[fid]    = f->N_eI;
        fun_Nee[fid]    = f->N_ee;
        fun_C[fid]      = f->C;
      }

  for (int iw = 0; iw < nw; ++iw)
  {
    auto& P                = p_list[iw];
    const auto& ee_table   = P.getDistTableAA(leader.ee_Table_ID_);
    const auto& eI_table   = P.getDistTableAB(leader.ei_Table_ID_);
    const auto& ee_r       = ee_table.getTempDists();
    const auto& ee_dr      = ee_table.getTempDispls();
    const auto& ei_r       = eI_table.getTempDists();
    const auto& ei_dr      = eI_table.getTempDispls();
    const auto& ei_full_r  = eI_table.getDistances();
    const auto& ei_full_dr = eI_table.getDisplacements();

    valT* ee_base = mem.mw_ee_temp.data() + static_cast<size_t>(iw) * (OHMMS_DIM + 1) * Nep;
    valT* ei_base = mem.mw_ei_temp.data() + static_cast<size_t>(iw) * (OHMMS_DIM + 1) * Nip;
    for (int k = 0; k < Ne; ++k)
      ee_base[k] = ee_r[k];
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
      for (int k = 0; k < Ne; ++k)
        ee_base[(idim + 1) * Nep + k] = ee_dr.data(idim)[k];
    for (int a = 0; a < Ni; ++a)
      ei_base[a] = ei_r[a];
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
      for (int a = 0; a < Ni; ++a)
        ei_base[(idim + 1) * Nip + a] = ei_dr.data(idim)[a];

    valT* full_r  = mem.mw_ei_full_r.data() + static_cast<size_t>(iw) * Ne * Nip;
    valT* full_dr = mem.mw_ei_full_dr.data() + static_cast<size_t>(iw) * OHMMS_DIM * Ne * Nip;
    for (int kel = 0; kel < Ne; ++kel)
      for (int a = 0; a < Ni; ++a)
      {
        full_r[kel * Nip + a] = ei_full_r[kel][a];
        for (int idim = 0; idim < OHMMS_DIM; ++idim)
          full_dr[idim * Ne * Nip + kel * Nip + a] = ei_full_dr[kel].data(idim)[a];
      }
  }
  mem.mw_ee_temp.updateTo();
  mem.mw_ei_temp.updateTo();
  mem.mw_ei_full_r.updateTo();
  mem.mw_ei_full_dr.updateTo();

  const size_t tb_bytes = sizeof(valT*) * nfun + sizeof(valT) * nfun + sizeof(int) * nfun * 3;
  mem.transfer_buffer.resize(tb_bytes);
  auto* tb          = mem.transfer_buffer.data();
  auto** gamma_ptrs = reinterpret_cast<valT**>(tb);
  auto* cut_ptr     = reinterpret_cast<valT*>(tb + sizeof(valT*) * nfun);
  auto* nei_ptr     = reinterpret_cast<int*>(tb + sizeof(valT*) * nfun + sizeof(valT) * nfun);
  auto* nee_ptr     = nei_ptr + nfun;
  auto* c_ptr       = nee_ptr + nfun;
  for (int f = 0; f < nfun; ++f)
  {
    gamma_ptrs[f] = const_cast<valT*>(gamma_host[f]);
    cut_ptr[f]    = fun_cut[f];
    nei_ptr[f]    = fun_NeI[f];
    nee_ptr[f]    = fun_Nee[f];
    c_ptr[f]      = fun_C[f];
  }

  auto* ee_ptr      = mem.mw_ee_temp.data();
  auto* ei_ptr      = mem.mw_ei_temp.data();
  auto* full_r_ptr  = mem.mw_ei_full_r.data();
  auto* full_dr_ptr = mem.mw_ei_full_dr.data();
  auto* vgl_ptr     = mem.mw_vgl.data();
  auto* Uk_ptr      = mem.mw_Uk.data();
  auto* dUk_ptr     = mem.mw_dUk.data();
  auto* d2Uk_ptr    = mem.mw_d2Uk.data();
  auto* egrp_ptr    = mem.e_grp.data();
  auto* igrp_ptr    = mem.i_grp.data();
  auto* cuti_ptr    = mem.ion_cutoff.data();
  auto* tb_ptr      = mem.transfer_buffer.data();
  const size_t tb_sz  = mem.transfer_buffer.size();
  const size_t ee_sz  = mem.mw_ee_temp.size();
  const size_t ei_sz  = mem.mw_ei_temp.size();
  const size_t fr_sz  = mem.mw_ei_full_r.size();
  const size_t fd_sz  = mem.mw_ei_full_dr.size();
  const size_t vgl_sz = static_cast<size_t>(nw) * (OHMMS_DIM + 2);
  const size_t Uk_sz  = mem.mw_Uk.size();
  const size_t dUk_sz = mem.mw_dUk.size();
  const size_t d2_sz  = mem.mw_d2Uk.size();

  PRAGMA_OFFLOAD("omp target teams distribute \
      map(to: ee_ptr[:ee_sz], ei_ptr[:ei_sz], full_r_ptr[:fr_sz], full_dr_ptr[:fd_sz]) \
      map(to: egrp_ptr[:Ne], igrp_ptr[:Ni], cuti_ptr[:Ni], tb_ptr[:tb_sz]) \
      map(from: vgl_ptr[:vgl_sz], Uk_ptr[:Uk_sz], dUk_ptr[:dUk_sz], d2Uk_ptr[:d2_sz])")
  for (int iw = 0; iw < nw; ++iw)
  {
    valT** gptrs = reinterpret_cast<valT**>(tb_ptr);
    valT* fcut   = reinterpret_cast<valT*>(tb_ptr + sizeof(valT*) * nfun);
    int* fNeI    = reinterpret_cast<int*>(tb_ptr + sizeof(valT*) * nfun + sizeof(valT) * nfun);
    int* fNee    = fNeI + nfun;
    int* fC      = fNee + nfun;

    const valT* ee_base = ee_ptr + iw * (OHMMS_DIM + 1) * Nep;
    const valT* ei_base = ei_ptr + iw * (OHMMS_DIM + 1) * Nip;
    const valT* fr      = full_r_ptr + iw * Ne * Nip;
    const valT* fdr     = full_dr_ptr + iw * OHMMS_DIM * Ne * Nip;

    valT Uj, d2Uj;
    valT dUj[3];
    jeei_offload::dense_ratio_grad_one_walker<valT>(iat, Ne, Nep, Ni, Nip, eG, iG, egrp_ptr, igrp_ptr, cuti_ptr,
                                                    ee_base, ee_base + Nep, ei_base, ei_base + Nip, fr, fdr,
                                                    const_cast<const valT**>(gptrs), fcut, fNeI, fNee, fC, Uj, dUj, d2Uj,
                                                    Uk_ptr + iw * Nep, dUk_ptr + iw * OHMMS_DIM * Nep,
                                                    d2Uk_ptr + iw * Nep);

    valT* vgl = vgl_ptr + iw * (OHMMS_DIM + 2);
    vgl[0]    = Uj;
    vgl[1]    = dUj[0];
    vgl[2]    = dUj[1];
    vgl[3]    = dUj[2];
    vgl[4]    = d2Uj;
  }

  for (int iw = 0; iw < nw; ++iw)
  {
    auto& wfc       = wfc_list.getCastedElement<JeeIOrbitalSoA<FT>>(iw);
    wfc.UpdateMode  = need_grad ? ORB_PBYP_PARTIAL : ORB_PBYP_RATIO;
    wfc.cur_Uat     = mem.mw_vgl[iw][0];
    wfc.cur_dUat[0] = mem.mw_vgl[iw][1];
    wfc.cur_dUat[1] = mem.mw_vgl[iw][2];
    wfc.cur_dUat[2] = mem.mw_vgl[iw][3];
    wfc.cur_d2Uat   = mem.mw_vgl[iw][4];
    for (int k = 0; k < Ne; ++k)
    {
      wfc.newUk[k]   = mem.mw_Uk[static_cast<size_t>(iw) * Nep + k];
      wfc.newd2Uk[k] = mem.mw_d2Uk[static_cast<size_t>(iw) * Nep + k];
    }
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
      for (int k = 0; k < Ne; ++k)
        wfc.newdUk.data(idim)[k] = mem.mw_dUk[static_cast<size_t>(iw) * OHMMS_DIM * Nep + idim * Nep + k];

    ratios[iw] = std::exp(static_cast<PsiValue>(wfc.Uat[iat] - wfc.cur_Uat));
    if (need_grad && grad_new)
      for (int idim = 0; idim < OHMMS_DIM; ++idim)
        (*grad_new)[iw][idim] += wfc.cur_dUat[idim];
  }
}

template<typename FT>
void JeeIOrbitalSoA<FT>::mw_ratioGrad_cuda(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                           const RefVectorWithLeader<ParticleSet>& p_list,
                                           int iat,
                                           std::vector<PsiValue>& ratios,
                                           std::vector<GradType>* grad_new,
                                           bool need_grad) const
{
#if !defined(ENABLE_CUDA)
  (void)wfc_list;
  (void)p_list;
  (void)iat;
  (void)ratios;
  (void)grad_new;
  (void)need_grad;
  throw std::runtime_error("mw_ratioGrad_cuda called without ENABLE_CUDA");
#else
  // Reuse host packing from mw_ratioGrad_offload memory layout, then launch CUDA.
  // Full precision (double) only — matches typical QMCPACK full-precision CUDA builds.
  static_assert(sizeof(valT) == sizeof(double), "JeeI CUDA path requires double RealType");
  auto& leader  = wfc_list.getCastedLeader<JeeIOrbitalSoA<FT>>();
  const int nw  = wfc_list.size();
  const int Ne  = leader.Nelec;
  const int Nep = static_cast<int>(leader.Nelec_padded);
  const int Ni  = leader.Nion;
  const int Nip = getAlignedSize<valT>(Ni);
  const int eG  = leader.eGroups;
  const int iG  = leader.iGroups;
  const int nfun = iG * eG * eG;

  JeeIOrbitalSoAMultiWalkerMem<RealType> local_mem;
  auto* mem_ptr = &local_mem;
  if (leader.mw_mem_handle_)
    mem_ptr = &leader.mw_mem_handle_.getResource();
  auto& mem = *mem_ptr;

  const size_t ee_temp_sz = static_cast<size_t>(nw) * (OHMMS_DIM + 1) * Nep;
  const size_t ei_temp_sz = static_cast<size_t>(nw) * (OHMMS_DIM + 1) * Nip;
  const bool temps_fresh  = mem.mw_ee_temp.size() != ee_temp_sz || mem.mw_ei_temp.size() != ei_temp_sz;
  mem.mw_vgl.resize(nw, OHMMS_DIM + 2);
  mem.mw_ee_temp.resize(ee_temp_sz);
  mem.mw_ei_temp.resize(ei_temp_sz);
  mem.mw_ei_full_r.resize(static_cast<size_t>(nw) * Ne * Nip);
  mem.mw_ei_full_dr.resize(static_cast<size_t>(nw) * OHMMS_DIM * Ne * Nip);
  mem.mw_Uk.resize(static_cast<size_t>(nw) * Nep);
  mem.mw_dUk.resize(static_cast<size_t>(nw) * OHMMS_DIM * Nep);
  mem.mw_d2Uk.resize(static_cast<size_t>(nw) * Nep);
  mem.e_grp.resize(Ne);
  mem.i_grp.resize(Ni);
  mem.ion_cutoff.resize(Ni);
  // Zero fresh temp buffers once: packing writes only [0, Ne)/[0, Ni) lanes, so pad
  // lanes stay deterministic zeros for the whole buffer lifetime.
  if (temps_fresh)
  {
    std::fill(mem.mw_ee_temp.begin(), mem.mw_ee_temp.end(), valT(0));
    std::fill(mem.mw_ei_temp.begin(), mem.mw_ei_temp.end(), valT(0));
  }

  auto& P0 = p_list[0];
  for (int e = 0; e < Ne; ++e)
    mem.e_grp[e] = P0.GroupID[e];
  for (int i = 0; i < Ni; ++i)
  {
    mem.i_grp[i]      = leader.Ions.GroupID[i];
    mem.ion_cutoff[i] = leader.Ion_cutoff[i];
  }

  // Hot path: build the static pack only once per resource; recompute (once per block)
  // refreshes it with force=true when parameters may have changed.
  refresh_cuda_static_pack(leader.F, iG, eG, mem, /*force=*/false);

  // Pack+H2D full e-I only when this resource's tables are marked dirty or the
  // thread-local workspace holds another owner's (or no) resident tables.
  const bool need_full = mem.cuda_full_dirty || !jeei_cuda::default_workspace().ownsFull(&mem);

#pragma omp parallel for schedule(static)
  for (int iw = 0; iw < nw; ++iw)
  {
    auto& P              = p_list[iw];
    const auto& ee_table = P.getDistTableAA(leader.ee_Table_ID_);
    const auto& eI_table = P.getDistTableAB(leader.ei_Table_ID_);
    const auto& ee_r     = ee_table.getTempDists();
    const auto& ee_dr    = ee_table.getTempDispls();
    const auto& ei_r     = eI_table.getTempDists();
    const auto& ei_dr    = eI_table.getTempDispls();

    valT* ee_base = mem.mw_ee_temp.data() + static_cast<size_t>(iw) * (OHMMS_DIM + 1) * Nep;
    valT* ei_base = mem.mw_ei_temp.data() + static_cast<size_t>(iw) * (OHMMS_DIM + 1) * Nip;
    for (int k = 0; k < Ne; ++k)
      ee_base[k] = ee_r[k];
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
      for (int k = 0; k < Ne; ++k)
        ee_base[(idim + 1) * Nep + k] = ee_dr.data(idim)[k];
    for (int a = 0; a < Ni; ++a)
      ei_base[a] = ei_r[a];
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
      for (int a = 0; a < Ni; ++a)
        ei_base[(idim + 1) * Nip + a] = ei_dr.data(idim)[a];

    if (need_full)
    {
      const auto& ei_full_r  = eI_table.getDistances();
      const auto& ei_full_dr = eI_table.getDisplacements();
      valT* full_r           = mem.mw_ei_full_r.data() + static_cast<size_t>(iw) * Ne * Nip;
      valT* full_dr          = mem.mw_ei_full_dr.data() + static_cast<size_t>(iw) * OHMMS_DIM * Ne * Nip;
      for (int kel = 0; kel < Ne; ++kel)
        for (int a = 0; a < Ni; ++a)
        {
          full_r[kel * Nip + a] = ei_full_r[kel][a];
          for (int idim = 0; idim < OHMMS_DIM; ++idim)
            full_dr[idim * Ne * Nip + kel * Nip + a] = ei_full_dr[kel].data(idim)[a];
        }
    }
  }

  jeei_cuda::launch_dense_ratio_grad(&mem, mem.cuda_static_version, iat, nw, Ne, Nep, Ni, Nip, eG, iG, nfun,
                                     mem.mw_ee_temp.data(),
                                     mem.mw_ei_temp.data(), mem.mw_ei_full_r.data(), mem.mw_ei_full_dr.data(),
                                     mem.e_grp.data(), mem.i_grp.data(), mem.ion_cutoff.data(),
                                     mem.cuda_gamma_pool.data(), mem.cuda_gamma_offset.data(),
                                     mem.cuda_gamma_len.data(), mem.cuda_fun_cut.data(), mem.cuda_fun_NeI.data(),
                                     mem.cuda_fun_Nee.data(), mem.cuda_fun_C.data(), need_full, mem.mw_vgl.data(),
                                     mem.mw_Uk.data(), mem.mw_dUk.data(), mem.mw_d2Uk.data());
  mem.cuda_full_dirty = false;

  for (int iw = 0; iw < nw; ++iw)
  {
    auto& wfc       = wfc_list.getCastedElement<JeeIOrbitalSoA<FT>>(iw);
    wfc.UpdateMode  = need_grad ? ORB_PBYP_PARTIAL : ORB_PBYP_RATIO;
    wfc.cur_Uat     = mem.mw_vgl[iw][0];
    wfc.cur_dUat[0] = mem.mw_vgl[iw][1];
    wfc.cur_dUat[1] = mem.mw_vgl[iw][2];
    wfc.cur_dUat[2] = mem.mw_vgl[iw][3];
    wfc.cur_d2Uat   = mem.mw_vgl[iw][4];
    for (int k = 0; k < Ne; ++k)
    {
      wfc.newUk[k]   = mem.mw_Uk[static_cast<size_t>(iw) * Nep + k];
      wfc.newd2Uk[k] = mem.mw_d2Uk[static_cast<size_t>(iw) * Nep + k];
    }
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
      for (int k = 0; k < Ne; ++k)
        wfc.newdUk.data(idim)[k] = mem.mw_dUk[static_cast<size_t>(iw) * OHMMS_DIM * Nep + idim * Nep + k];

    ratios[iw] = std::exp(static_cast<PsiValue>(wfc.Uat[iat] - wfc.cur_Uat));
    if (need_grad && grad_new)
      for (int idim = 0; idim < OHMMS_DIM; ++idim)
        (*grad_new)[iw][idim] += wfc.cur_dUat[idim];
  }
#endif
}

template<typename FT>
void JeeIOrbitalSoA<FT>::mw_calcRatio(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                      const RefVectorWithLeader<ParticleSet>& p_list,
                                      int iat,
                                      std::vector<PsiValue>& ratios) const
{
  assert(this == &wfc_list.getLeader());
#if defined(ENABLE_CUDA)
  if (use_offload_)
  {
    mw_ratioGrad_cuda(wfc_list, p_list, iat, ratios, nullptr, false);
    return;
  }
#endif
#if defined(ENABLE_OFFLOAD)
  if (use_offload_)
  {
    mw_ratioGrad_offload(wfc_list, p_list, iat, ratios, nullptr, false);
    return;
  }
#endif
  // Host multi-walker: serial per-move loop; walker parallelism comes from crowds,
  // and a parallel region per move costs more than these evaluations.
  const int nw = wfc_list.size();
  for (int iw = 0; iw < nw; iw++)
    ratios[iw] = wfc_list[iw].ratio(p_list[iw], iat);
}

template<typename FT>
void JeeIOrbitalSoA<FT>::mw_ratioGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                      const RefVectorWithLeader<ParticleSet>& p_list,
                                      int iat,
                                      std::vector<PsiValue>& ratios,
                                      std::vector<GradType>& grad_new) const
{
  assert(this == &wfc_list.getLeader());
#if defined(ENABLE_CUDA)
  if (use_offload_)
  {
    mw_ratioGrad_cuda(wfc_list, p_list, iat, ratios, &grad_new, true);
    return;
  }
#endif
#if defined(ENABLE_OFFLOAD)
  if (use_offload_)
  {
    mw_ratioGrad_offload(wfc_list, p_list, iat, ratios, &grad_new, true);
    return;
  }
#endif
  // Host multi-walker: serial per-move loop; walker parallelism comes from crowds,
  // and a parallel region per move costs more than these evaluations.
  const int nw = wfc_list.size();
  for (int iw = 0; iw < nw; iw++)
    ratios[iw] = wfc_list[iw].ratioGrad(p_list[iw], iat, grad_new[iw]);
}

template<typename FT>
void JeeIOrbitalSoA<FT>::mw_accept_rejectMove(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                              const RefVectorWithLeader<ParticleSet>& p_list,
                                              int iat,
                                              const std::vector<bool>& isAccepted,
                                              bool safe_to_delay) const
{
  assert(this == &wfc_list.getLeader());
  const int nw = wfc_list.size();
#pragma omp parallel for schedule(static)
  for (int iw = 0; iw < nw; iw++)
  {
    if (!isAccepted[iw])
      continue;
    wfc_list[iw].acceptMove(p_list[iw], iat, safe_to_delay);
  }

  if (use_offload_)
  {
    auto& wfc_leader = wfc_list.getCastedLeader<JeeIOrbitalSoA<FT>>();
    if (wfc_leader.mw_mem_handle_)
    {
      auto& mem = wfc_leader.mw_mem_handle_.getResource();
      mem.mw_allUat.updateTo();
#if defined(ENABLE_CUDA)
      auto& ws = jeei_cuda::default_workspace();
      if (ws.ownsFull(&mem))
      {
        // Patch accepted walkers' e-I rows on device so temps-only ratioGrad stays valid.
        // This component's accept runs before the ParticleSet accept, so getDistRow(iat)
        // still holds the pre-move row; the accepted position lives in the temp row.
        const int Ne        = wfc_leader.Nelec;
        const int Ni        = wfc_leader.Nion;
        const int Nip       = getAlignedSize<valT>(Ni);
        std::vector<double> r_row(Nip, 0.0), dr_row(3 * Nip, 0.0);
        for (int iw = 0; iw < nw; ++iw)
        {
          if (!isAccepted[iw])
            continue;
          const auto& eI_table = p_list[iw].getDistTableAB(wfc_leader.ei_Table_ID_);
          const auto& r        = eI_table.getTempDists();
          const auto& dr       = eI_table.getTempDispls();
          for (int a = 0; a < Ni; ++a)
            r_row[a] = static_cast<double>(r[a]);
          for (int idim = 0; idim < OHMMS_DIM; ++idim)
            for (int a = 0; a < Ni; ++a)
              dr_row[idim * Nip + a] = static_cast<double>(dr.data(idim)[a]);
          ws.updateEiRow(&mem, iw, iat, Ne, Nip, Ni, r_row.data(), dr_row.data());
        }
      }
      else
      {
        // Resident tables belong to another owner (or none): repack on next ratioGrad.
        mem.cuda_full_dirty = true;
      }
#else
      mem.cuda_full_dirty = true;
#endif
    }
  }
}

template<typename FT>
void JeeIOrbitalSoA<FT>::mw_evaluateGL(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                       const RefVectorWithLeader<ParticleSet>& p_list,
                                       const RefVector<ParticleSet::ParticleGradient>& G_list,
                                       const RefVector<ParticleSet::ParticleLaplacian>& L_list,
                                       bool fromscratch) const
{
  assert(this == &wfc_list.getLeader());
  const int nw = wfc_list.size();
  if (fromscratch)
  {
    const std::vector<bool> recompute_all(nw, true);
    mw_recompute(wfc_list, p_list, recompute_all);
  }
  for (int iw = 0; iw < nw; iw++)
  {
    auto& wfc      = wfc_list.getCastedElement<JeeIOrbitalSoA<FT>>(iw);
    wfc.log_value_ = wfc.computeGL(G_list[iw], L_list[iw]);
  }
}

template<typename FT>
void JeeIOrbitalSoA<FT>::mw_evaluateRatios(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                           const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
                                           std::vector<std::vector<ValueType>>& ratios) const
{
  assert(this == &wfc_list.getLeader());
  const int nw = wfc_list.size();
  for (int iw = 0; iw < nw; iw++)
    wfc_list[iw].evaluateRatios(vp_list[iw], ratios[iw]);
}

template class JeeIOrbitalSoA<PolynomialFunctor3D>;

} // namespace qmcplusplus
