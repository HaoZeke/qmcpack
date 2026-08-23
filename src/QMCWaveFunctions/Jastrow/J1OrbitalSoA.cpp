//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-


#include <type_traits>
#include <cstdlib>
#include <iostream>
#include "J1OrbitalSoA.h"
#include "SoaDistanceTableABOMPTarget.h"
#include "ResourceCollection.h"

namespace qmcplusplus
{

namespace
{
/** ask the table for the batch's temporary distances on the device and report readiness
 *
 * The table produces them for a consumer that has asked, because the kernel that makes
 * them grows with the number of sources and buys nothing on its own. The batch that asks
 * first finds them absent and reads the host distances for that move.
 */
/** the work below which a launch costs more than the host reduction it replaces
 *
 * A two ion cell at 64 walkers measures 5.1 percent slower, 2 of 8 pairs, which puts the
 * crossover near that product. Every batched entry point here asks the same question, so
 * they ask it in one place and cannot drift apart.
 *
 * The value is overridable because a caller forming its factor from what the batched form
 * left on the device depends on that form having run, and a small case otherwise has no way
 * to be driven down the batched path deliberately.
 *
 * Lowering it is a performance choice only. It once also held out a two ion cell that came
 * back with a NaN kinetic energy, from the electron-ion table reporting device temp distances
 * ready before they were written; that is fixed in the table, so the batched path is correct
 * at any work and this is free to tune.
 */
size_t batchedWorkThreshold()
{
  static const size_t value = [] {
    if (const char* c = std::getenv("QMCPACK_J1_BATCH_MIN_WORK"))
      if (const long v = std::atol(c); v >= 0)
        return static_cast<size_t>(v);
    return size_t(512);
  }();
  return value;
}

bool deviceTempDistancesReady(const ParticleSet& p_leader, int table_id)
{
  const auto& dt = p_leader.getDistTableAB(table_id);
  if (dt.hasTempDataOnDevice())
    return true;
  dt.requireTempDataOnDevice();
  return false;
}
} // namespace

/** whether a functor offers the batched value and gradient form
 *
 * Only some of the functors a one-body Jastrow is instantiated for provide it, so the
 * batched reduction is taken where it exists and the base loop serves the rest.
 */
template<typename T, typename = void>
struct HasMwEvaluateVGL : std::false_type
{};
template<typename T>
struct HasMwEvaluateVGL<T, std::void_t<decltype(&T::mw_evaluateVGL)>> : std::true_type
{};


template<typename T>
struct J1OrbitalSoAMultiWalkerMem : public Resource
{
  // fused buffer for fast transfer
  Vector<char, OffloadPinnedAllocator<char>> transfer_buffer;
  // multi walker result
  Vector<T, OffloadPinnedAllocator<T>> mw_vals;
  // multi walker -1
  Vector<int, OffloadPinnedAllocator<int>> mw_minus_one;
  // multi walker value and gradient for a proposed move, [nw][DIM+2]
  Matrix<T, OffloadPinnedAllocator<T>> mw_vgl;
  // the current value at the moved electron, one per walker, for the device factor
  Vector<T, OffloadPinnedAllocator<T>> mw_vat;
  // per source scratch the value and gradient kernel writes, [nw][3][n_padded]
  Vector<T, OffloadPinnedAllocator<T>> mw_cur_allu;
  // fused buffer for the value and gradient kernel
  Vector<char, OffloadPinnedAllocator<char>> mw_ratiograd_buffer;

  void resize_minus_one(size_t size)
  {
    if (mw_minus_one.size() < size)
    {
      mw_minus_one.resize(size, -1);
      mw_minus_one.updateTo();
    }
  }

  J1OrbitalSoAMultiWalkerMem() : Resource("J1OrbitalSoAMultiWalkerMem") {}

  J1OrbitalSoAMultiWalkerMem(const J1OrbitalSoAMultiWalkerMem&) : J1OrbitalSoAMultiWalkerMem() {}

  std::unique_ptr<Resource> makeClone() const override { return std::make_unique<J1OrbitalSoAMultiWalkerMem>(*this); }
};

template<typename FT>
J1OrbitalSoA<FT>::J1OrbitalSoA(const std::string& obj_name, const ParticleSet& ions, ParticleSet& els, bool use_offload)
    : WaveFunctionComponent(obj_name),
      use_offload_(use_offload),
      myTableID(els.addTable(ions, use_offload ? DTModes::ALL_OFF : DTModes::NEED_VP_FULL_TABLE_ON_HOST)),
      Nions(ions.getTotalNum()),
      Nelec(els.getTotalNum()),
      NumGroups(ions.groups()),
      Ions(ions)
{
  if (my_name_.empty())
    throw std::runtime_error("J1OrbitalSoA object name cannot be empty!");

  if (use_offload_)
    assert(ions.getCoordinates().getKind() == DynamicCoordinateKind::DC_POS_OFFLOAD);

  initialize(els);

  // set up grp_ids
  grp_ids.resize(Nions);
  int count = 0;
  for (int ig = 0; ig < NumGroups; ig++)
    for (int j = ions.first(ig); j < ions.last(ig); j++)
      grp_ids[count++] = ig;
  assert(count == Nions);
  grp_ids.updateTo();
}

template<typename FT>
J1OrbitalSoA<FT>::~J1OrbitalSoA() = default;

template<typename FT>
void J1OrbitalSoA<FT>::checkSanity() const
{
  if (std::any_of(J1Functors.begin(), J1Functors.end(), [](auto* ptr) { return ptr == nullptr; }))
    app_warning() << "One-body Jastrow \"" << my_name_ << "\" doesn't cover all the particle pairs. "
                  << "Consider fusing multiple entries if they are of the same type for optimal code performance."
                  << std::endl;
}

template<typename FT>
void J1OrbitalSoA<FT>::createResource(ResourceCollection& collection) const
{ collection.addResource(std::make_unique<J1OrbitalSoAMultiWalkerMem<RealType>>()); }

template<typename FT>
void J1OrbitalSoA<FT>::acquireResource(ResourceCollection& collection,
                                       const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  auto& wfc_leader          = wfc_list.getCastedLeader<J1OrbitalSoA<FT>>();
  wfc_leader.mw_mem_handle_ = collection.lendResource<J1OrbitalSoAMultiWalkerMem<RealType>>();
}

template<typename FT>
void J1OrbitalSoA<FT>::releaseResource(ResourceCollection& collection,
                                       const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  auto& wfc_leader = wfc_list.getCastedLeader<J1OrbitalSoA<FT>>();
  collection.takebackResource(wfc_leader.mw_mem_handle_);
}

template<typename FT>
void J1OrbitalSoA<FT>::mw_accept_rejectMove(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                            const RefVectorWithLeader<ParticleSet>& p_list,
                                            int iat,
                                            const std::vector<bool>& isAccepted,
                                            bool safe_to_delay) const
{
  assert(this == &wfc_list.getLeader());
  auto& wfc_leader = wfc_list.getCastedLeader<J1OrbitalSoA<FT>>();
  const int nw     = wfc_list.size();

  /* After a ratio-only move the single walker accept recomputes the reduction over ions,
   * because ratio() did not need the gradient and laplacian that the accept stores. That
   * recompute is per walker on the host, and with a high acceptance it undoes most of
   * what batching the ratio saved, so it is batched under the same gate.
   *
   * A move that came through ratioGrad already has the values and needs no recompute,
   * which is what the update mode says.
   */
  const bool needs_recompute = wfc_leader.UpdateMode == ORB_PBYP_RATIO;
  if constexpr (HasMwEvaluateVGL<FT>::value)
  {
    if (needs_recompute && use_offload_ && static_cast<size_t>(nw) * wfc_leader.Nions >= batchedWorkThreshold() &&
        deviceTempDistancesReady(p_list.getLeader(), wfc_leader.myTableID))
    {
      auto& p_leader        = p_list.getLeader();
      const auto& dt_leader = p_leader.getDistTableAB(wfc_leader.myTableID);
      auto& mw_mem          = wfc_leader.mw_mem_handle_.getResource();
      auto& mw_vgl          = mw_mem.mw_vgl;
      auto& mw_cur_allu     = mw_mem.mw_cur_allu;
      const size_t n_padded = getAlignedSize<valT>(wfc_leader.Nions);
      mw_vgl.resize(nw, DIM + 2);
      mw_cur_allu.resize(n_padded * 3 * nw);

      FT::mw_evaluateVGL(-1, NumGroups, GroupFunctors.data(), wfc_leader.Nions, grp_ids.data(), nw, mw_vgl.device_data(),
                         n_padded, dt_leader.getMultiWalkerTempDataPtr(), mw_cur_allu.data(),
                         mw_mem.mw_ratiograd_buffer);
  mw_vgl.updateFrom(); // read on the host just below

      for (int iw = 0; iw < nw; iw++)
      {
        auto& wfc  = wfc_list.getCastedElement<J1OrbitalSoA<FT>>(iw);
        wfc.curAt  = mw_vgl[iw][0];
        wfc.curLap = -mw_vgl[iw][DIM + 1];
        for (int idim = 0; idim < DIM; idim++)
          wfc.curGrad[idim] = mw_vgl[iw][idim + 1];
        // the values are in hand, so the single walker accept must not redo them
        wfc.UpdateMode = ORB_PBYP_PARTIAL;
      }
    }
  }

  for (int iw = 0; iw < nw; iw++)
    if (isAccepted[iw])
      wfc_list[iw].acceptMove(p_list[iw], iat, safe_to_delay);
    else
      wfc_list[iw].restore(iat);
}

template<typename FT>
void J1OrbitalSoA<FT>::mw_calcRatio(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                    const RefVectorWithLeader<ParticleSet>& p_list,
                                    int iat,
                                    std::vector<PsiValue>& ratios) const
{
  assert(this == &wfc_list.getLeader());
  auto& wfc_leader = wfc_list.getCastedLeader<J1OrbitalSoA<FT>>();
  const int nw     = wfc_list.size();

  /* Same trade as mw_ratioGrad, and the same gate: a launch against the host's reduction
   * over ions for every walker.
   */
  if (!use_offload_ || static_cast<size_t>(nw) * wfc_leader.Nions < batchedWorkThreshold() ||
      !deviceTempDistancesReady(p_list.getLeader(), wfc_leader.myTableID))
  {
    WaveFunctionComponent::mw_calcRatio(wfc_list, p_list, iat, ratios);
    return;
  }

  auto& p_leader        = p_list.getLeader();
  const auto& dt_leader = p_leader.getDistTableAB(wfc_leader.myTableID);
  auto& mw_mem          = wfc_leader.mw_mem_handle_.getResource();
  auto& mw_vals         = mw_mem.mw_vals;
  const size_t n_padded = getAlignedSize<valT>(wfc_leader.Nions);

  mw_vals.resize(nw);
  mw_mem.resize_minus_one(nw);

  // the value only form of what mw_ratioGrad uses, over the moved electron's distances
  FT::mw_evaluateV(NumGroups, GroupFunctors.data(), wfc_leader.Nions, grp_ids.data(), nw, mw_mem.mw_minus_one.data(),
                   dt_leader.getMultiWalkerTempDataPtr(), n_padded * (DIM + 1), mw_vals.data(),
                   mw_mem.transfer_buffer);

  for (int iw = 0; iw < nw; iw++)
  {
    auto& wfc      = wfc_list.getCastedElement<J1OrbitalSoA<FT>>(iw);
    wfc.UpdateMode = ORB_PBYP_RATIO;
    wfc.curAt      = mw_vals[iw];
    ratios[iw]     = std::exp(static_cast<PsiValue>(wfc.Vat[iat] - wfc.curAt));
  }
}

template<typename FT>
void J1OrbitalSoA<FT>::mw_ratioGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                    const RefVectorWithLeader<ParticleSet>& p_list,
                                    int iat,
                                    std::vector<PsiValue>& ratios,
                                    std::vector<GradType>& grad_new) const
{
  if constexpr (!HasMwEvaluateVGL<FT>::value)
  {
    WaveFunctionComponent::mw_ratioGrad(wfc_list, p_list, iat, ratios, grad_new);
    return;
  }
  else
  {
  if (!use_offload_)
  {
    WaveFunctionComponent::mw_ratioGrad(wfc_list, p_list, iat, ratios, grad_new);
    return;
  }

  /* The single walker form reduces over the ions on the host, once per walker. The same
   * reduction for the whole batch is one kernel, the one the two-body Jastrow already
   * uses, given the moved electron's distances to every ion on the device. Nothing here
   * is excluded from the sum, unlike the two-body case, so the index it skips is set past
   * the end.
   */
  assert(this == &wfc_list.getLeader());
  auto& wfc_leader      = wfc_list.getCastedLeader<J1OrbitalSoA<FT>>();
  auto& p_leader        = p_list.getLeader();
  const auto& dt_leader = p_leader.getDistTableAB(wfc_leader.myTableID);
  const int nw          = wfc_list.size();

  /* The batched form trades a pair of kernel launches per move for the host's reduction over
   * ions, once per walker. Below some amount of work the launches cost more than the loop
   * they replace; batchedWorkThreshold carries where that sits and why.
   */
  if (static_cast<size_t>(nw) * wfc_leader.Nions < batchedWorkThreshold() ||
      !deviceTempDistancesReady(p_list.getLeader(), wfc_leader.myTableID))
  {
    WaveFunctionComponent::mw_ratioGrad(wfc_list, p_list, iat, ratios, grad_new);
    return;
  }

  auto& mw_mem      = wfc_leader.mw_mem_handle_.getResource();
  auto& mw_vgl      = mw_mem.mw_vgl;
  auto& mw_cur_allu = mw_mem.mw_cur_allu;
  const size_t n_padded = getAlignedSize<valT>(wfc_leader.Nions);
  mw_vgl.resize(nw, DIM + 2);
  mw_cur_allu.resize(n_padded * 3 * nw);

  FT::mw_evaluateVGL(-1, NumGroups, GroupFunctors.data(), wfc_leader.Nions, grp_ids.data(), nw, mw_vgl.device_data(),
                     n_padded, dt_leader.getMultiWalkerTempDataPtr(), mw_cur_allu.data(),
                     mw_mem.mw_ratiograd_buffer);
  mw_vgl.updateFrom(); // read on the host just below

  for (int iw = 0; iw < nw; iw++)
  {
    auto& wfc = wfc_list.getCastedElement<J1OrbitalSoA<FT>>(iw);
    wfc.UpdateMode = ORB_PBYP_PARTIAL;
    wfc.curAt      = mw_vgl[iw][0];
    // the kernel stores the negated laplacian, which is what the two-body path wants;
    // accumulateGL returns it unnegated, so this matches the single walker form
    wfc.curLap     = -mw_vgl[iw][DIM + 1];
    for (int idim = 0; idim < DIM; idim++)
      wfc.curGrad[idim] = mw_vgl[iw][idim + 1];
    ratios[iw] = std::exp(static_cast<PsiValue>(wfc.Vat[iat] - wfc.curAt));
    grad_new[iw] += wfc.curGrad;
  }
  }
}

template<typename FT>
void J1OrbitalSoA<FT>::mw_ratioGradDevice(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                          const RefVectorWithLeader<ParticleSet>& p_list,
                                          int iat,
                                          std::vector<PsiValue>& ratios,
                                          std::vector<GradType>& grad_new,
                                          Vector<PsiValue, OffloadPinnedAllocator<PsiValue>>& ratios_device_prod,
                                          Vector<ValueType, OffloadPinnedAllocator<ValueType>>& grads_device_sum) const
{
  if constexpr (!HasMwEvaluateVGL<FT>::value)
  {
    WaveFunctionComponent::mw_ratioGradDevice(wfc_list, p_list, iat, ratios, grad_new, ratios_device_prod,
                                              grads_device_sum);
    return;
  }
  else
  {
  assert(this == &wfc_list.getLeader());
  auto& wfc_leader = wfc_list.getCastedLeader<J1OrbitalSoA<FT>>();
  const int nw     = wfc_list.size();

  /* The factor is formed from mw_vgl, which carries the proposed value and gradient on the
   * device only when the batched form ran there. That form decides on the work available and
   * on the table having temp distances resident, so the same question is asked here, the same
   * way, rather than assumed. Where it does not run, the base form stages its own results,
   * which is two transfers against the one below.
   */
  if (!wfc_leader.use_offload_ || static_cast<size_t>(nw) * wfc_leader.Nions < batchedWorkThreshold() ||
      !deviceTempDistancesReady(p_list.getLeader(), wfc_leader.myTableID))
  {
    WaveFunctionComponent::mw_ratioGradDevice(wfc_list, p_list, iat, ratios, grad_new, ratios_device_prod,
                                              grads_device_sum);
    return;
  }

  mw_ratioGrad(wfc_list, p_list, iat, ratios, grad_new);

  /* Vat at the moved electron is the only part of the ratio the device does not already hold,
   * so it is the only thing that goes down: one scalar per walker.
   */
  auto& mw_mem = wfc_leader.mw_mem_handle_.getResource();
  auto& mw_vgl = mw_mem.mw_vgl;
  auto& mw_vat = mw_mem.mw_vat;
  mw_vat.resize(nw);
  for (int iw = 0; iw < nw; iw++)
    mw_vat[iw] = wfc_list.getCastedElement<J1OrbitalSoA<FT>>(iw).Vat[iat];
  mw_vat.updateTo();

  const size_t vstr   = mw_vgl.cols();
  const auto* vgl_ptr = mw_vgl.device_data();
  const auto* vat_ptr = mw_vat.device_data();
  auto* rd_ptr        = ratios_device_prod.device_data();
  auto* gs_ptr        = grads_device_sum.device_data();
  constexpr int dim   = OHMMS_DIM;

  PRAGMA_OFFLOAD("omp target teams distribute parallel for is_device_ptr(vgl_ptr, vat_ptr, rd_ptr, gs_ptr)")
  for (int iw = 0; iw < nw; iw++)
  {
    rd_ptr[iw] *= static_cast<PsiValue>(std::exp(vat_ptr[iw] - vgl_ptr[iw * vstr]));
    for (int id = 0; id < dim; id++)
      gs_ptr[iw * dim + id] += static_cast<ValueType>(vgl_ptr[iw * vstr + id + 1]);
  }
  }
}

template<typename FT>
void J1OrbitalSoA<FT>::mw_evaluateRatios(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                         const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
                                         std::vector<std::vector<ValueType>>& ratios) const
{
  if (!use_offload_)
  {
    WaveFunctionComponent::mw_evaluateRatios(wfc_list, vp_list, ratios);
    return;
  }

  // add early return to prevent from accessing vp_list[0]
  if (wfc_list.size() == 0)
    return;
  auto& wfc_leader        = wfc_list.getCastedLeader<J1OrbitalSoA<FT>>();
  auto& vp_leader         = vp_list.getLeader();
  const auto& mw_refPctls = vp_leader.getMultiWalkerRefPctls();
  auto& mw_mem            = wfc_leader.mw_mem_handle_.getResource();
  auto& mw_vals           = mw_mem.mw_vals;
  auto& mw_minus_one      = mw_mem.mw_minus_one;
  const int nw            = wfc_list.size();

  const size_t nVPs = mw_refPctls.size();
  mw_vals.resize(nVPs);
  mw_mem.resize_minus_one(nVPs);

  const auto& dt_leader(vp_leader.getDistTableAB(wfc_leader.myTableID));

  FT::mw_evaluateV(NumGroups, GroupFunctors.data(), wfc_leader.Nions, grp_ids.data(), nVPs, mw_minus_one.data(),
                   dt_leader.getMultiWalkerDataPtr(), dt_leader.getPerTargetPctlStrideSize(), mw_vals.data(),
                   mw_mem.transfer_buffer);

  size_t ivp = 0;
  for (int iw = 0; iw < nw; ++iw)
  {
    const VirtualParticleSet& vp = vp_list[iw];
    auto& wfc                    = wfc_list.getCastedElement<J1OrbitalSoA<FT>>(iw);
    for (int k = 0; k < vp.getTotalNum(); ++k, ivp++)
      ratios[iw][k] = std::exp(wfc.Vat[mw_refPctls[ivp]] - mw_vals[ivp]);
  }
  assert(ivp == nVPs);
}

template class J1OrbitalSoA<BsplineFunctor<QMCTraits::RealType>>;
template class J1OrbitalSoA<
    CubicSplineSingle<QMCTraits::RealType, CubicBspline<QMCTraits::RealType, LINEAR_1DGRID, FIRSTDERIV_CONSTRAINTS>>>;
template class J1OrbitalSoA<UserFunctor<QMCTraits::RealType>>;
template class J1OrbitalSoA<ShortRangeCuspFunctor<QMCTraits::RealType>>;
template class J1OrbitalSoA<PadeFunctor<QMCTraits::RealType>>;
template class J1OrbitalSoA<Pade2ndOrderFunctor<QMCTraits::RealType>>;
} // namespace qmcplusplus
