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
  // per source scratch the value and gradient kernel writes, [nw][3][n_padded]
  Vector<T, OffloadPinnedAllocator<T>> mw_cur_allu;
  // fused buffer for the value and gradient kernel
  Vector<char, OffloadPinnedAllocator<char>> mw_ratiograd_buffer;
  /* memory pool for Vat, Grad and Lap across the crowd, [nw][n_padded] + [nw][n_padded][DIM]
   * + [nw][n_padded]. Each walker's own containers are views into it, so the accept can
   * write the stored state where the kernels read it.
   */
  Vector<T, OffloadPinnedAllocator<T>> mw_allVat;
  // the log value change the accept kernel forms, one per walker
  Vector<T, OffloadPinnedAllocator<T>> mw_log_delta;
  // the stored gradient at one electron, gathered for a host reader, [nw][DIM]
  Vector<T, OffloadPinnedAllocator<T>> mw_grad_at;
  // the accepted walker list the accept kernel branches on, or the mask's own indices
  Vector<int, OffloadPinnedAllocator<int>> mw_accepted;
  /* whether the last ratio call left mw_vgl on the device. The batched ratio runs under a
   * work threshold and a residency test, and the accept kernel reads what it wrote, so the
   * accept asks this rather than repeating the test and drifting from it.
   */
  bool mw_vgl_on_device = false;

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
      Nelec_padded(getAlignedSize<valT>(els.getTotalNum())),
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

  const size_t nw   = wfc_list.size();
  const size_t npad = wfc_leader.Nelec_padded;
  auto& mw_allVat   = wfc_leader.mw_mem_handle_.getResource().mw_allVat;
  mw_allVat.resize(npad * (OHMMS_DIM + 2) * nw);
  for (size_t iw = 0; iw < nw; iw++)
  {
    // copy each walker's Vat, Grad and Lap into the shared buffer and attach to it
    auto& wfc = wfc_list.getCastedElement<J1OrbitalSoA<FT>>(iw);

    valT* vat_ptr = mw_allVat.data() + iw * npad;
    std::copy_n(wfc.Vat.data(), wfc.Nelec, vat_ptr);
    wfc.Vat.free();
    wfc.Vat.attachReference(vat_ptr, wfc.Nelec);

    posT* grad_ptr = reinterpret_cast<posT*>(mw_allVat.data() + nw * npad + iw * npad * OHMMS_DIM);
    std::copy_n(wfc.Grad.data(), wfc.Nelec, grad_ptr);
    wfc.Grad.free();
    wfc.Grad.attachReference(grad_ptr, wfc.Nelec);

    valT* lap_ptr = mw_allVat.data() + nw * npad * (OHMMS_DIM + 1) + iw * npad;
    std::copy_n(wfc.Lap.data(), wfc.Nelec, lap_ptr);
    wfc.Lap.free();
    wfc.Lap.attachReference(lap_ptr, wfc.Nelec);
  }
  mw_allVat.updateTo();
}

template<typename FT>
void J1OrbitalSoA<FT>::releaseResource(ResourceCollection& collection,
                                       const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  auto& wfc_leader  = wfc_list.getCastedLeader<J1OrbitalSoA<FT>>();
  const size_t nw   = wfc_list.size();
  const size_t npad = wfc_leader.Nelec_padded;
  auto& mw_allVat   = wfc_leader.mw_mem_handle_.getResource().mw_allVat;
  mw_allVat.updateFrom();
  for (size_t iw = 0; iw < nw; iw++)
  {
    // detach and give each walker its own storage back, carrying the values out
    auto& wfc = wfc_list.getCastedElement<J1OrbitalSoA<FT>>(iw);

    const valT* vat_ptr  = mw_allVat.data() + iw * npad;
    const posT* grad_ptr = reinterpret_cast<const posT*>(mw_allVat.data() + nw * npad + iw * npad * OHMMS_DIM);
    const valT* lap_ptr  = mw_allVat.data() + nw * npad * (OHMMS_DIM + 1) + iw * npad;

    wfc.Vat.free();
    wfc.Vat.resize(wfc.Nelec);
    std::copy_n(vat_ptr, wfc.Nelec, wfc.Vat.data());

    wfc.Grad.free();
    wfc.Grad.resize(wfc.Nelec);
    std::copy_n(grad_ptr, wfc.Nelec, wfc.Grad.data());

    wfc.Lap.free();
    wfc.Lap.resize(wfc.Nelec);
    std::copy_n(lap_ptr, wfc.Nelec, wfc.Lap.data());
  }
  collection.takebackResource(wfc_leader.mw_mem_handle_);
}

template<typename FT>
void J1OrbitalSoA<FT>::mw_accept_rejectMove(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                            const RefVectorWithLeader<ParticleSet>& p_list,
                                            int iat,
                                            const std::vector<bool>& isAccepted,
                                            bool safe_to_delay) const
{
  syncHostState(wfc_list);
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

  /* The accept writes three numbers at the moved electron and moves the log value by the
   * difference between the stored value there and the proposed one. All four come from
   * mw_vgl, which the ratio kernel left on the device, and the state they land in is
   * device resident too, so the whole thing is one kernel over the crowd and nw scalars
   * come back. The host form is a per walker call that reads and writes the same state
   * on the host, which is what makes the state's residency the thing that decides.
   */
  if constexpr (HasMwEvaluateVGL<FT>::value)
  {
    if (use_offload_ && wfc_leader.UpdateMode != ORB_PBYP_RATIO &&
        wfc_leader.mw_mem_handle_.getResource().mw_vgl_on_device)
    {
      auto& mw_mem       = wfc_leader.mw_mem_handle_.getResource();
      auto& mw_log_delta = mw_mem.mw_log_delta;
      auto& mw_accepted  = mw_mem.mw_accepted;
      mw_log_delta.resize(nw);
      mw_accepted.resize(nw);
      for (int iw = 0; iw < nw; iw++)
        mw_accepted[iw] = isAccepted[iw] ? 1 : 0;
      mw_accepted.updateTo();

      const size_t npad   = wfc_leader.Nelec_padded;
      constexpr int dim   = OHMMS_DIM;
      const size_t vstr   = mw_mem.mw_vgl.cols();
      const auto* vgl_ptr = mw_mem.mw_vgl.device_data();
      auto* vat_ptr       = mw_mem.mw_allVat.device_data();
      const auto* acc_ptr = mw_accepted.device_data();
      auto* delta_ptr     = mw_log_delta.device_data();
      const size_t nwz    = nw;

      PRAGMA_OFFLOAD("omp target teams distribute parallel for \
                      is_device_ptr(vgl_ptr, vat_ptr, acc_ptr, delta_ptr)")
      for (size_t iw = 0; iw < nwz; iw++)
      {
        if (!acc_ptr[iw])
        {
          delta_ptr[iw] = valT(0);
          continue;
        }
        valT* Vat  = vat_ptr + iw * npad;
        valT* Grad = vat_ptr + nwz * npad + iw * npad * dim;
        valT* Lap  = vat_ptr + nwz * npad * (dim + 1) + iw * npad;

        const valT* vgl = vgl_ptr + iw * vstr;
        delta_ptr[iw]   = Vat[iat] - vgl[0];
        Vat[iat]        = vgl[0];
        for (int id = 0; id < dim; id++)
          Grad[iat * dim + id] = vgl[id + 1];
        // the kernel stores the negated laplacian, as the host form's curLap does
        Lap[iat] = -vgl[dim + 1];
      }
      mw_log_delta.updateFrom();

      for (int iw = 0; iw < nw; iw++)
        wfc_list.getCastedElement<J1OrbitalSoA<FT>>(iw).log_value_ += mw_log_delta[iw];
      return;
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
  syncHostState(wfc_list);
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
  syncHostState(wfc_list);
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
    wfc_leader.mw_mem_handle_.getResource().mw_vgl_on_device = false;
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
  mw_mem.mw_vgl_on_device = true;
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
                                          Vector<ValueType, OffloadPinnedAllocator<ValueType>>& grads_device_sum,
                                          bool assign) const
{
  syncHostState(wfc_list);
  if constexpr (!HasMwEvaluateVGL<FT>::value)
  {
    WaveFunctionComponent::mw_ratioGradDevice(wfc_list, p_list, iat, ratios, grad_new, ratios_device_prod,
                                              grads_device_sum, assign);
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
                                              grads_device_sum, assign);
    return;
  }

  /* Vat and the proposed value are both device resident, so the factor is a difference and
   * an exponential away from being formed where they are, and nothing here reads either on
   * the host. The proposed values still have to be computed, which is the kernel below;
   * what this form drops against mw_ratioGrad is the copy down of mw_vgl and the per walker
   * host loop over it. ratios and grad_new are left untouched, as the caller's contract says.
   */
  auto& p_leader        = p_list.getLeader();
  const auto& dt_leader = p_leader.getDistTableAB(wfc_leader.myTableID);
  auto& mw_mem          = wfc_leader.mw_mem_handle_.getResource();
  auto& mw_vgl          = mw_mem.mw_vgl;
  const size_t n_padded = getAlignedSize<valT>(wfc_leader.Nions);
  mw_vgl.resize(nw, DIM + 2);
  mw_mem.mw_cur_allu.resize(n_padded * 3 * nw);

  FT::mw_evaluateVGL(-1, NumGroups, GroupFunctors.data(), wfc_leader.Nions, grp_ids.data(), nw, mw_vgl.device_data(),
                     n_padded, dt_leader.getMultiWalkerTempDataPtr(), mw_mem.mw_cur_allu.data(),
                     mw_mem.mw_ratiograd_buffer);
  mw_mem.mw_vgl_on_device = true;
  for (int iw = 0; iw < nw; iw++)
    wfc_list.getCastedElement<J1OrbitalSoA<FT>>(iw).UpdateMode = ORB_PBYP_PARTIAL;
  wfc_leader.UpdateMode = ORB_PBYP_PARTIAL;

  const size_t npad   = wfc_leader.Nelec_padded;
  const size_t vstr   = mw_vgl.cols();
  const size_t nwz    = nw;
  const auto* vgl_ptr = mw_vgl.device_data();
  const auto* vat_ptr = mw_mem.mw_allVat.device_data();
  auto* rd_ptr        = ratios_device_prod.device_data();
  auto* gs_ptr        = grads_device_sum.device_data();
  constexpr int dim   = OHMMS_DIM;

  PRAGMA_OFFLOAD("omp target teams distribute parallel for is_device_ptr(vgl_ptr, vat_ptr, rd_ptr, gs_ptr)")
  for (size_t iw = 0; iw < nwz; iw++)
  {
    const auto factor = static_cast<PsiValue>(std::exp(vat_ptr[iw * npad + iat] - vgl_ptr[iw * vstr]));
    if (assign)
    {
      rd_ptr[iw] = factor;
      for (int id = 0; id < dim; id++)
        gs_ptr[iw * dim + id] = static_cast<ValueType>(vgl_ptr[iw * vstr + id + 1]);
    }
    else
    {
      rd_ptr[iw] *= factor;
      for (int id = 0; id < dim; id++)
        gs_ptr[iw * dim + id] += static_cast<ValueType>(vgl_ptr[iw * vstr + id + 1]);
    }
  }
  }
}

template<typename FT>
void J1OrbitalSoA<FT>::syncHostState(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list)
{
  if (wfc_list.size() == 0)
    return;
  auto& wfc_leader = wfc_list.getCastedLeader<J1OrbitalSoA<FT>>();
  if (!wfc_leader.use_offload_)
    return;
  bool dirty = false;
  for (int iw = 0; iw < wfc_list.size(); iw++)
  {
    auto& wfc = wfc_list.getCastedElement<J1OrbitalSoA<FT>>(iw);
    dirty |= wfc.host_state_dirty_;
    wfc.host_state_dirty_ = false;
  }
  /* Every walker's host copy is current when one of them goes dirty, because the reader
   * that ran before the single walker write fetched the whole buffer, so the push is the
   * whole buffer too.
   */
  if (dirty)
    wfc_leader.mw_mem_handle_.getResource().mw_allVat.updateTo();
}

template<typename FT>
void J1OrbitalSoA<FT>::mw_recompute(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                    const RefVectorWithLeader<ParticleSet>& p_list,
                                    const std::vector<bool>& recompute) const
{
  WaveFunctionComponent::mw_recompute(wfc_list, p_list, recompute);
  if (use_offload_)
    wfc_list.getCastedLeader<J1OrbitalSoA<FT>>().mw_mem_handle_.getResource().mw_allVat.updateTo();
}

template<typename FT>
void J1OrbitalSoA<FT>::mw_evaluateLog(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                      const RefVectorWithLeader<ParticleSet>& p_list,
                                      const RefVector<ParticleSet::ParticleGradient>& G_list,
                                      const RefVector<ParticleSet::ParticleLaplacian>& L_list) const
{
  WaveFunctionComponent::mw_evaluateLog(wfc_list, p_list, G_list, L_list);
  if (use_offload_)
    wfc_list.getCastedLeader<J1OrbitalSoA<FT>>().mw_mem_handle_.getResource().mw_allVat.updateTo();
}

template<typename FT>
void J1OrbitalSoA<FT>::mw_evalGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                   const RefVectorWithLeader<ParticleSet>& p_list,
                                   int iat,
                                   std::vector<GradType>& grad_now) const
{
  syncHostState(wfc_list);
  if (!use_offload_)
  {
    WaveFunctionComponent::mw_evalGrad(wfc_list, p_list, iat, grad_now);
    return;
  }

  auto& wfc_leader  = wfc_list.getCastedLeader<J1OrbitalSoA<FT>>();
  const int nw      = wfc_list.size();
  auto& mw_mem      = wfc_leader.mw_mem_handle_.getResource();
  auto& mw_grad_at  = mw_mem.mw_grad_at;
  constexpr int dim = OHMMS_DIM;
  mw_grad_at.resize(nw * dim);

  /* The state is [nw][n_padded][DIM] after the values, and a host reader wants nw times DIM
   * of it, so the gather is what goes down rather than the array.
   */
  {
    const size_t npad   = wfc_leader.Nelec_padded;
    const size_t nwz    = nw;
    const auto* vat_ptr = mw_mem.mw_allVat.device_data();
    auto* out_dev       = mw_grad_at.device_data();
    PRAGMA_OFFLOAD("omp target teams distribute parallel for is_device_ptr(vat_ptr, out_dev)")
    for (size_t iw = 0; iw < nwz; iw++)
      for (int id = 0; id < dim; id++)
        out_dev[iw * dim + id] = vat_ptr[nwz * npad + iw * npad * dim + static_cast<size_t>(iat) * dim + id];
  }
  mw_grad_at.updateFrom();

  // the caller accumulates across components, so this assigns its own term
  for (int iw = 0; iw < nw; iw++)
    for (int id = 0; id < dim; id++)
      grad_now[iw][id] = mw_grad_at[iw * dim + id];
}

template<typename FT>
void J1OrbitalSoA<FT>::mw_evalGradDevice(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                         const RefVectorWithLeader<ParticleSet>& p_list,
                                         int iat,
                                         std::vector<GradType>& grad_now,
                                         Vector<ValueType, OffloadPinnedAllocator<ValueType>>& grads_device_now,
                                         bool assign) const
{
  syncHostState(wfc_list);
  if (!use_offload_)
  {
    WaveFunctionComponent::mw_evalGradDevice(wfc_list, p_list, iat, grad_now, grads_device_now, assign);
    return;
  }

  auto& wfc_leader  = wfc_list.getCastedLeader<J1OrbitalSoA<FT>>();
  const int nw      = wfc_list.size();
  const size_t npad = wfc_leader.Nelec_padded;
  constexpr int dim = OHMMS_DIM;
  const size_t nwz  = nw;

  const auto* vat_ptr = wfc_leader.mw_mem_handle_.getResource().mw_allVat.device_data();
  auto* dst_ptr       = grads_device_now.device_data();

  PRAGMA_OFFLOAD("omp target teams distribute parallel for is_device_ptr(vat_ptr, dst_ptr)")
  for (size_t iw = 0; iw < nwz; iw++)
    for (int id = 0; id < dim; id++)
    {
      const auto term =
          static_cast<ValueType>(vat_ptr[nwz * npad + iw * npad * dim + static_cast<size_t>(iat) * dim + id]);
      if (assign)
        dst_ptr[iw * dim + id] = term;
      else
        dst_ptr[iw * dim + id] += term;
    }
}

template<typename FT>
void J1OrbitalSoA<FT>::mw_evaluateGL(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                     const RefVectorWithLeader<ParticleSet>& p_list,
                                     const RefVector<ParticleSet::ParticleGradient>& G_list,
                                     const RefVector<ParticleSet::ParticleLaplacian>& L_list,
                                     bool fromscratch) const
{
  syncHostState(wfc_list);
  if (use_offload_)
    // computeGL sums the host Vat, Grad and Lap, which the accept path leaves on the device
    wfc_list.getCastedLeader<J1OrbitalSoA<FT>>().mw_mem_handle_.getResource().mw_allVat.updateFrom();
  WaveFunctionComponent::mw_evaluateGL(wfc_list, p_list, G_list, L_list, fromscratch);
}

template<typename FT>
void J1OrbitalSoA<FT>::mw_evaluateRatios(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                         const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
                                         std::vector<std::vector<ValueType>>& ratios) const
{
  syncHostState(wfc_list);
  if (!use_offload_)
  {
    WaveFunctionComponent::mw_evaluateRatios(wfc_list, vp_list, ratios);
    return;
  }

  // add early return to prevent from accessing vp_list[0]
  if (wfc_list.size() == 0)
    return;
  auto& wfc_leader        = wfc_list.getCastedLeader<J1OrbitalSoA<FT>>();

  /* Vat stays on the device across the electron loop, and this is one of the two readers
   * that bring it back. Both run once per step rather than once per electron.
   */
  wfc_leader.mw_mem_handle_.getResource().mw_allVat.updateFrom();

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
