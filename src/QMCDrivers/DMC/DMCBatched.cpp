//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File refactored from: DMC.cpp
//////////////////////////////////////////////////////////////////////////////////////

#include <functional>
#include <cassert>
#include <cmath>

#include "DMCBatched.h"
#include "QMCDrivers/GreenFunctionModifiers/DriftModifierBase.h"
#include "QMCDrivers/GreenFunctionModifiers/DriftModifierUNR.h"
#include "Concurrency/ParallelExecutor.hpp"
#include "Concurrency/Info.hpp"
#include "Message/UniformCommunicateError.h"
#include "Message/CommOperators.h"
#include "ParticleBase/RandomSeqGenerator.h"
#include "Utilities/RunTimeManager.h"
#include "Utilities/ProgressReportEngine.h"
#include "QMCDrivers/DMC/WalkerControl.h"
#include "QMCDrivers/SFNBranch.h"
#include <PSdispatcher.h>
#include <TWFdispatcher.h>
#include <Hdispatcher.h>
#include "EstimatorInputDelegates.h"
#include "MemoryUsage.h"
#include "QMCWaveFunctions/TWFGrads.hpp"
#include "TauParams.hpp"
#include "WalkerLogManager.h"
#include "CPU/math.hpp"
#include "DMCContextForSteps.h"

namespace qmcplusplus
{
/** The Metropolis test for one electron, on the device.
 *
 *  Reads the ratio and the two Green function exponents, applies the same three predicates
 *  the host form applies and compares against the pre-drawn variate, writing the acceptance
 *  mask where the accept paths can read it. Nothing here comes back to the host, which is
 *  the point: the mask is the last value the per-electron step made host visible.
 *
 *  The arithmetic mirrors the host expression exactly, including the epsilon floor on prob,
 *  so with the same inputs it produces the same decision.
 */
template<typename RT, typename PsiV>
inline void dmcAcceptanceOnDevice(size_t nw,
                                  const PsiV* ratios,
                                  const RT* log_gf,
                                  const RT* log_gb,
                                  const char* are_valid,
                                  const RT* variates,
                                  char* accepted)
{
  // accepted is a device address: the mask is what the accept path reads, so it stays there and
  // the host takes a copy only if it needs one for its own bookkeeping

  PRAGMA_OFFLOAD("omp target teams distribute parallel for \
                  map(always, to: ratios[0:nw], log_gf[0:nw], log_gb[0:nw], are_valid[0:nw], \
                                  variates[0:nw]) \
                  is_device_ptr(accepted)")
  for (size_t iw = 0; iw < nw; iw++)
  {
    const RT prob = std::norm(ratios[iw]) * std::exp(log_gb[iw] - log_gf[iw]);
    accepted[iw]  = (are_valid[iw] != 0 && ratios[iw] != PsiV(0) && prob >= std::numeric_limits<RT>::epsilon() &&
                    variates[iw] < prob)
        ? 1
        : 0;
  }
}

/** The whole per-electron decision, on the device, in one kernel.
 *
 *  The host form is five loops: scale the gradient into a reverse drift, form the two Green's
 *  function exponents, form the probability, test the phase, compare against the variate. Each
 *  reads values the device already holds, so each is a reason to bring them over. Done here they
 *  are one launch and the ratios and gradients never leave the device.
 *
 *  The arithmetic is the host's, term for term: driftScalingUNR and logGreensFunctionPos are the
 *  same functions the host path calls, declared target so both sides compute the same value, and
 *  the variates are the ones drawn for the step, consumed in the same order.
 *
 *  packed carries what the host still owns for this electron: the forward drift and the
 *  displacement, [nw][dim] each, then the validity flags, so it is one transfer rather than three.
 */
template<typename RT, typename PsiV, typename VT>
inline void dmcDecisionOnDevice(size_t nw,
                                int dim,
                                RT tau,
                                RT oneover2tau,
                                RT drift_a,
                                const PsiV* ratios_dev,
                                const VT* grads_dev,
                                const RT* packed,
                                const RT* variates,
                                char* accepted)
{
  const size_t drift_off = 0;
  const size_t delta_off = nw * dim;
  const size_t valid_off = 2 * nw * dim;
  PRAGMA_OFFLOAD("omp target teams distribute parallel for \
                  map(always, to: packed[0:2 * nw * dim + nw]) \
                  is_device_ptr(ratios_dev, grads_dev, variates, accepted)")
  for (size_t iw = 0; iw < nw; iw++)
  {
    RT vsq(0);
    for (int id = 0; id < dim; id++)
    {
      const RT g = std::real(grads_dev[iw * dim + id]);
      vsq += g * g;
    }
    const RT sc = driftScalingUNR(tau, drift_a, vsq);

    RT gb_sq(0), gf_sq(0);
    for (int id = 0; id < dim; id++)
    {
      // drifts_reverse = scaled gradient, then the forward drift added, as the host does
      const RT rev = std::real(grads_dev[iw * dim + id]) * sc + packed[drift_off + iw * dim + id];
      gb_sq += rev * rev;
      const RT d = packed[delta_off + iw * dim + id];
      gf_sq += d * d;
    }
    const RT log_gb = -oneover2tau * gb_sq;
    const RT log_gf = -oneover2tau * gf_sq;

    const PsiV ratio = ratios_dev[iw];
    const RT prob    = std::norm(ratio) * std::exp(log_gb - log_gf);
    const bool valid = packed[valid_off + iw] != RT(0);
    accepted[iw]     = (valid && ratio != PsiV(0) && prob >= std::numeric_limits<RT>::epsilon() &&
                    variates[iw] < prob)
        ? 1
        : 0;
  }
}

using std::placeholders::_1;
using WP       = WalkerProperties::Indexes;
using PsiValue = TrialWaveFunction::PsiValue;

/** Constructor maintains proper ownership of input parameters
 *
 *  Note you must call the Base constructor before the derived class sets QMCType
 */
DMCBatched::DMCBatched(const ProjectData& project_data,
                       QMCDriverInput&& qmcdriver_input,
                       UPtr<EstimatorManagerNew>&& estimator_manager,
                       DMCDriverInput&& input,
                       WalkerConfigurations& wc,
                       MCPopulation&& pop,
                       const RefVector<RandomBase<FullPrecRealType>>& rng_refs,
                       Communicate* comm)
    : QMCDriverNew(project_data,
                   std::move(qmcdriver_input),
                   std::move(estimator_manager),
                   wc,
                   std::move(pop),
                   rng_refs,
                   "DMCBatched::",
                   comm,
                   "DMCBatched"),
      dmcdriver_input_(input),
      dmc_timers_("DMCBatched::")
{}

DMCBatched::~DMCBatched() = default;

template<CoordsType CT>
void DMCBatched::advanceWalkers(const StateForThread& sft,
                                Crowd& crowd,
                                DriverTimers& timers,
                                DMCTimers& dmc_timers,
                                DMCContextForSteps& step_context,
                                bool recompute,
                                bool accumulate_this_step)
{
  const PSdispatcher ps_dispatcher(!sft.serializing_crowd_walkers);
  const TWFdispatcher twf_dispatcher(!sft.serializing_crowd_walkers);
  const Hdispatcher ham_dispatcher(!sft.serializing_crowd_walkers);

  auto& walkers = crowd.get_walkers();
  const RefVectorWithLeader<ParticleSet> walker_elecs(crowd.get_walker_elecs()[0], crowd.get_walker_elecs());
  const RefVectorWithLeader<TrialWaveFunction> walker_twfs(crowd.get_walker_twfs()[0], crowd.get_walker_twfs());
  const RefVectorWithLeader<QMCHamiltonian> walker_hamiltonians(crowd.get_walker_hamiltonians()[0],
                                                                crowd.get_walker_hamiltonians());

  timers.resource_timer.start();
  ResourceCollectionTeamLock<ParticleSet> pset_res_lock(crowd.getSharedResource().pset_res, walker_elecs);
  ResourceCollectionTeamLock<TrialWaveFunction> twfs_res_lock(crowd.getSharedResource().twf_res, walker_twfs);
  ResourceCollectionTeamLock<QMCHamiltonian> hams_res_lock(crowd.getSharedResource().ham_res, walker_hamiltonians);
  timers.resource_timer.stop();

  {
    ScopedTimer recompute_timer(dmc_timers.step_begin_recompute_timer);
    std::vector<bool> recompute_mask;
    recompute_mask.reserve(walkers.size());
    for (MCPWalker& awalker : walkers)
      if (awalker.wasTouched)
      {
        recompute_mask.push_back(true);
        awalker.wasTouched = false;
      }
      else
        recompute_mask.push_back(false);
    ps_dispatcher.flex_loadWalker(walker_elecs, walkers, recompute_mask, true);
    twf_dispatcher.flex_recompute(walker_twfs, walker_elecs, recompute_mask);
  }

  const int num_walkers   = crowd.size();
  auto& pset_leader       = walker_elecs.getLeader();
  const int num_particles = pset_leader.getTotalNum();

  std::vector<bool> are_valid(num_walkers);
  MCCoords<CT> drifts(num_walkers), drifts_reverse(num_walkers);
  MCCoords<CT> walker_deltas(num_walkers * num_particles), deltas(num_walkers);
  TWFGrads<CT> grads_now(num_walkers), grads_new(num_walkers);

  //This generates an entire steps worth of deltas.
  makeGaussRandomWithEngine(walker_deltas, step_context.get_random_gen());

  // The acceptance variates for the whole step, drawn here for the same reason the
  // displacements are: a device side acceptance test cannot call a host generator, and an
  // array can be uploaded once instead. Drawn for every walker and particle rather than
  // only where the earlier tests pass, so the count does not depend on the data.
  Vector<RealType, OffloadPinnedAllocator<RealType>> accept_rands(num_walkers * num_particles);
  Vector<char, OffloadPinnedAllocator<char>> dev_valid, dev_accepted;
  size_t device_accept_mismatches = 0;
  Vector<PsiValue, OffloadPinnedAllocator<PsiValue>> device_ratio_prod;
  Vector<QMCTraits::ValueType, OffloadPinnedAllocator<QMCTraits::ValueType>> device_grad_sum;
  /// the gradient at the current position and the drift formed from it, both device side
  Vector<QMCTraits::ValueType, OffloadPinnedAllocator<QMCTraits::ValueType>> device_grad_now;
  Vector<RealType, OffloadPinnedAllocator<RealType>> device_drifts;
  /// forward drift, displacement and validity for one electron, packed into one transfer
  Vector<RealType, OffloadPinnedAllocator<RealType>> decision_packed;

  /* Whether the decision can be taken on the device at all, decided once rather than per electron.
   *
   * Serialized crowd walkers put the dispatchers on their single walker forms, which a device mask
   * has nothing to say to. The device form also reproduces the drift scaling itself, since the
   * modifier's own entry point is virtual and cannot be called from a target region, so it may only
   * run for a modifier whose scaling it knows.
   */
  const bool device_decision_possible = [&sft] {
    if constexpr (CT != CoordsType::POS)
      return false; // spin coordinates carry a second Green's function term the kernel does not form
#if !defined(QMC_COMPLEX)
    return false; // a real build rejects on a phase change, which the kernel does not test
#else
    if (sft.serializing_crowd_walkers || !sft.drift_modifier.isUNRScaling())
      return false;
    const char* d = std::getenv("QMCPACK_DMC_DEVICE_ACCEPT");
    return d && *d == '1';
#endif
  }();
  /* Whether the gradient, the drift and the proposal are formed on the device.
   *
   * Separate from the decision because the chain does not yet end there: the move needs a
   * host position for the lattice validity test and for the distance tables' per walker
   * temp arrays, so the drift is read back, and the fold kernels that formed it are paid
   * for nothing. Measured at 134 us per electron against the host prologue on one device.
   * It stays exercisable, and switching it on is what the move's own residency
   * (qmcpack-vve0) has to be measured against.
   */
  const bool device_prologue = [device_decision_possible] {
    if (!device_decision_possible)
      return false;
    const char* d = std::getenv("QMCPACK_DMC_DEVICE_DRIFT");
    return d && *d == '1';
  }();
  std::vector<PsiValue> dev_ratios;
  std::vector<TrialWaveFunction::GradType> dev_grads;
  /// host values for the device cross-check, kept apart from the ones the step uses
  TWFGrads<CT> chk_grads(num_walkers);
  std::vector<PsiValue> chk_ratios(num_walkers);
  size_t device_ratio_mismatches = 0;
  size_t device_grad_mismatches  = 0;
  for (size_t i = 0; i < accept_rands.size(); i++)
    accept_rands[i] = step_context.get_random_gen()();
  accept_rands.updateTo();

  std::vector<PsiValue> ratios(num_walkers, PsiValue(0.0));
  std::vector<RealType> log_gf(num_walkers, 0.0);
  std::vector<RealType> log_gb(num_walkers, 0.0);
  std::vector<RealType> prob(num_walkers, 0.0);

  // local list to handle accept/reject
  std::vector<bool> isAccepted;
  isAccepted.reserve(num_walkers);

  //save the old energies for branching needs.
  std::vector<FullPrecRealType> old_energies(num_walkers);
  for (int iw = 0; iw < num_walkers; ++iw)
    old_energies[iw] = walkers[iw].get().Properties(WP::LOCALENERGY);

  std::vector<RealType> rr_proposed(num_walkers, 0.0);
  std::vector<RealType> rr_accepted(num_walkers, 0.0);
  // per-particle scratch, reused across the particle loop
  std::vector<RealType> rr(num_walkers, 0.0);
  std::vector<int> rejects(num_walkers);

  {
    ScopedTimer pbyp_local_timer(timers.movepbyp_timer);
    for (int ig = 0; ig < pset_leader.groups(); ++ig)
    {
      TauParams<RealType, CT> taus(sft.qmcdrv_input.get_tau(), 1.0 / pset_leader.get_mass_by_group()[ig],
                                   sft.qmcdrv_input.get_spin_mass());

      twf_dispatcher.flex_prepareGroup(walker_twfs, walker_elecs, ig);

      for (int iat = pset_leader.first(ig); iat < pset_leader.last(ig); ++iat)
      {
        //This is very useful thing to be able to look at in the debugger
#ifndef NDEBUG
        std::vector<int> walkers_who_have_been_on_wire(num_walkers, 0);
        for (int iw = 0; iw < walkers.size(); ++iw)
        {
          walkers[iw].get().get_has_been_on_wire() ? walkers_who_have_been_on_wire[iw] = 1
                                                   : walkers_who_have_been_on_wire[iw] = 0;
        }
#endif
        //get deltas for this particle for all walkers
        walker_deltas.getSubset(iat * num_walkers, num_walkers, deltas);

        // only DMC does this
        // TODO: rr needs a real name
        // Reused across the particle loop rather than allocated inside it: this body
        // runs once per particle per step, so a fresh vector here is one heap
        // allocation per particle per step per crowd.
        rr.resize(num_walkers);
        assert(rr.size() == deltas.positions.size());
        std::transform(deltas.positions.begin(), deltas.positions.end(), rr.begin(),
                       [t = taus.tauovermass](auto& delta_r) { return t * dot(delta_r, delta_r); });

        /* The gradient, the drift and the proposal it feeds are all one chain, and the
         * device form keeps the gradient and the drift there: the components add their
         * terms into device_grad_now and the modifier scales it into device_drifts,
         * neither of which the host forms. The host form copies each result down between
         * the steps.
         */
        if (device_prologue)
        {
          constexpr int dim = QMCTraits::DIM;
          device_drifts.resize(num_walkers * dim);
          TrialWaveFunction::mw_evalGradDevice(walker_twfs, walker_elecs, iat, grads_now.grads_positions,
                                               device_grad_now);
          sft.drift_modifier.getDriftsDevice(taus.tauovermass, num_walkers, dim, device_grad_now.device_data(),
                                             device_drifts.device_data());
          /* The drift is read back because the move still needs a host position: the
           * lattice validity test and the distance tables' per walker temp arrays are
           * both host side, and a component whose accept reads those arrays depends on
           * them. Once that state is device resident this copy goes and the proposal is
           * formed by mw_makeActivePosOnDevice from device_drifts directly.
           */
          device_drifts.updateFrom();
          for (int iw = 0; iw < num_walkers; iw++)
            for (int id = 0; id < dim; id++)
              drifts.positions[iw][id] = device_drifts[iw * dim + id];
          scaleBySqrtTau(taus, deltas);
          drifts += deltas;
        }
        else
        {
          twf_dispatcher.flex_evalGrad(walker_twfs, walker_elecs, iat, grads_now);
          sft.drift_modifier.getDrifts(taus, grads_now, drifts);

          scaleBySqrtTau(taus, deltas);
          drifts += deltas;
        }

// in DMC this was done here, changed to match VMCBatched pending factoring to common source
// if (rr > m_r2max)
// {
//   ++nRejectTemp;
//   continue;
// }
#ifndef NDEBUG
        for (int i = 0; i < rr.size(); ++i)
          assert(qmcplusplus::isfinite(rr[i]));
#endif

        ps_dispatcher.flex_makeMove(walker_elecs, iat, drifts, are_valid);

        if (device_decision_possible)
          TrialWaveFunction::mw_calcRatioGradDevice(walker_twfs, walker_elecs, iat, ratios, dev_grads,
                                                    device_ratio_prod, device_grad_sum);
        else
          twf_dispatcher.flex_calcRatioGrad(walker_twfs, walker_elecs, iat, ratios, grads_new);

        /* Cross-check the device product against the host one before anything relies on
         * it. The device form recomputes the same component ratios and multiplies them
         * where they already are, so a disagreement is a bug in that path rather than
         * something to average away. Spinor coordinates carry a second gradient the
         * device form does not produce, so they are left out.
         */
        if constexpr (CT == CoordsType::POS)
          if (const char* d = std::getenv("QMCPACK_CHECK_DEVICE_RATIO"); d && *d == '1')
          {
            /* The host values have to be computed here rather than read from ratios and
             * grads_new. On this path mw_calcRatioGradDevice poisons ratios, and a
             * comparison against a NaN is false however wrong the device value is, so
             * reading it makes the check pass without testing anything. grads_new is not
             * written on this path at all, so reading it compares against another
             * electron's gradient and fails however right the device value is.
             */
            twf_dispatcher.flex_calcRatioGrad(walker_twfs, walker_elecs, iat, chk_ratios, chk_grads);
            device_ratio_prod.updateFrom();
            device_grad_sum.updateFrom();
            for (int iw = 0; iw < num_walkers; iw++)
            {
              const RealType mag  = std::abs(chk_ratios[iw]);
              const RealType diff = std::abs(device_ratio_prod[iw] - chk_ratios[iw]);
              if (diff > RealType(1e-9) * std::max(mag, RealType(1)))
                device_ratio_mismatches++;
              for (int id = 0; id < QMCTraits::DIM; id++)
              {
                const RealType gmag = std::abs(chk_grads.grads_positions[iw][id]);
                const RealType gdiff =
                    std::abs(device_grad_sum[iw * QMCTraits::DIM + id] - chk_grads.grads_positions[iw][id]);
                if (gdiff > RealType(1e-9) * std::max(gmag, RealType(1)))
                  device_grad_mismatches++;
              }
            }
            /* The host form above left the components' state as it computes it, and the
             * accept below reads the device form's. Running it again restores that.
             */
            TrialWaveFunction::mw_calcRatioGradDevice(walker_twfs, walker_elecs, iat, ratios, dev_grads,
                                                      device_ratio_prod, device_grad_sum);
          }

        if (!device_decision_possible)
        {
          // the device form folds all of this into its own kernel, from values already there
          computeLogGreensFunction(deltas, taus, log_gf);

          sft.drift_modifier.getDrifts(taus, grads_new, drifts_reverse);

          drifts_reverse += drifts;

          computeLogGreensFunction(drifts_reverse, taus, log_gb);
        }

        auto checkPhaseChanged = [&sft](const PsiValue& ratio, int& is_reject) {
          if (ratio == PsiValue(0) || sft.branch_engine.phaseChanged(std::arg(ratio)))
            is_reject = 1;
          else
            is_reject = 0;
        };

        // Hopefully a phase change doesn't make any of these transformations fail.
        // reused rather than allocated per particle, as with rr above
        rejects.resize(num_walkers); // instead of std::vector<bool>
        for (int iw = 0; iw < num_walkers; ++iw)
        {
          // a complex build never reports a phase change, and rr_proposed is host bookkeeping
          // that the decision does not enter
          if (!device_decision_possible)
            checkPhaseChanged(ratios[iw], rejects[iw]);
          else
            rejects[iw] = 0;
          rr_proposed[iw] += rr[iw];
        }

        if (!device_decision_possible)
          for (int iw = 0; iw < num_walkers; ++iw)
            prob[iw] = std::norm(ratios[iw]) * std::exp(log_gb[iw] - log_gf[iw]);

        // Cross-check the device form against the host one before it replaces it: the kernel
        // has to reproduce this decision exactly for every walker, and a disagreement is a
        // bug in the kernel's arithmetic rather than something to average away.
        if (const char* d = std::getenv("QMCPACK_CHECK_DEVICE_ACCEPT"); d && *d == '1')
        {
          dev_valid.resize(num_walkers);
          dev_accepted.resize(num_walkers);
          for (int iw = 0; iw < num_walkers; ++iw)
            dev_valid[iw] = (are_valid[iw] && !rejects[iw]) ? 1 : 0;
          // the mask is written where the accept path reads it, so the comparison takes a copy
          dmcAcceptanceOnDevice<RealType, PsiValue>(num_walkers, ratios.data(), log_gf.data(), log_gb.data(),
                                                    dev_valid.data(), accept_rands.data() + iat * num_walkers,
                                                    dev_accepted.device_data());
          dev_accepted.updateFrom();
          for (int iw = 0; iw < num_walkers; ++iw)
          {
            const bool host_accept = are_valid[iw] && !rejects[iw] &&
                prob[iw] >= std::numeric_limits<RealType>::epsilon() &&
                accept_rands[iat * num_walkers + iw] < prob[iw];
            if (host_accept != (dev_accepted[iw] != 0))
              device_accept_mismatches++;
          }
        }

        /* The decision can be made where the values already are. DMC draws its variates for the
         * whole step up front, so the device form consumes exactly the same numbers in the same
         * order as the host form and reaches the same answer, which is what makes this switchable
         * rather than a different sampling. VMC cannot do this as it stands: its host form draws
         * inside a short-circuited condition, so the sequence depends on the data.
         *
         * The mask stays on the device for the accept path. The host still needs the decision for
         * its own counters and for rr_accepted, so it takes one copy of nw bytes, once, rather
         * than the ratios and gradients it would otherwise have formed the decision from.
         */
        const bool device_accept = device_decision_possible;

        if (device_accept)
        {
          /* One kernel for the whole decision, reading the ratios and gradients where they were
           * computed. What the host still owns for this electron, the forward drift and the
           * displacement and the validity flags, goes down packed as one transfer.
           */
          constexpr int dim = QMCTraits::DIM;
          dev_accepted.resize(num_walkers);
          decision_packed.resize(2 * num_walkers * dim + num_walkers);
          auto* pk = decision_packed.data();
          for (int iw = 0; iw < num_walkers; ++iw)
            for (int id = 0; id < dim; id++)
            {
              pk[iw * dim + id]                      = drifts.positions[iw][id];
              pk[num_walkers * dim + iw * dim + id]  = deltas.positions[iw][id];
            }
          for (int iw = 0; iw < num_walkers; ++iw)
            pk[2 * num_walkers * dim + iw] = (are_valid[iw] && !rejects[iw]) ? RealType(1) : RealType(0);

          dmcDecisionOnDevice<RealType, PsiValue, QMCTraits::ValueType>(num_walkers, dim, taus.tauovermass,
                                                             taus.oneover2tau,
                                                             sft.drift_modifier.getUNRScalingA(),
                                                             device_ratio_prod.device_data(),
                                                             device_grad_sum.device_data(), pk,
                                                             accept_rands.device_data() + iat * num_walkers,
                                                             dev_accepted.device_data());
          dev_accepted.updateFrom();

          isAccepted.clear();
          for (int iw = 0; iw < num_walkers; ++iw)
            if (dev_accepted[iw])
            {
              crowd.incAccept();
              isAccepted.push_back(true);
              rr_accepted[iw] += rr[iw];
            }
            else
            {
              crowd.incReject();
              isAccepted.push_back(false);
            }

          TrialWaveFunction::mw_accept_rejectMoveFromDeviceMask(walker_twfs, walker_elecs, iat,
                                                                dev_accepted.device_data(), true);
          ps_dispatcher.flex_accept_rejectMove<CT>(walker_elecs, iat, isAccepted);
        }
        else
        {
        isAccepted.clear();

        for (int iw = 0; iw < num_walkers; ++iw)
          if (are_valid[iw] && !rejects[iw] && prob[iw] >= std::numeric_limits<RealType>::epsilon() &&
              accept_rands[iat * num_walkers + iw] < prob[iw])
          {
            crowd.incAccept();
            isAccepted.push_back(true);
            rr_accepted[iw] += rr[iw];
          }
          else
          {
            crowd.incReject();
            isAccepted.push_back(false);
          }

        twf_dispatcher.flex_accept_rejectMove(walker_twfs, walker_elecs, iat, isAccepted, true);

        ps_dispatcher.flex_accept_rejectMove<CT>(walker_elecs, iat, isAccepted);
        }
      }
    }

    twf_dispatcher.flex_completeUpdates(walker_twfs);
    ps_dispatcher.flex_donePbyP(walker_elecs);
  }

  if (const char* d = std::getenv("QMCPACK_CHECK_DEVICE_ACCEPT"); d && *d == '1')
    std::cerr << "DEVACCEPT mismatches=" << device_accept_mismatches << " over " << num_particles << " electrons and "
              << num_walkers << " walkers" << std::endl;

  if (const char* d = std::getenv("QMCPACK_CHECK_DEVICE_RATIO"); d && *d == '1')
    std::cerr << "DEVRATIO mismatches=" << device_ratio_mismatches << " grad_mismatches=" << device_grad_mismatches
              << " over " << num_particles << " electrons and " << num_walkers << " walkers" << std::endl;

  { // collect GL for KE.
    ScopedTimer buffer_local(timers.buffer_timer);
    twf_dispatcher.flex_evaluateGL(walker_twfs, walker_elecs, recompute);
    if (sft.qmcdrv_input.get_debug_checks() & DriverDebugChecks::CHECKGL_AFTER_MOVES)
      checkLogAndGL(crowd, "checkGL_after_moves", sft.serializing_crowd_walkers);
    ps_dispatcher.flex_saveWalker(walker_elecs, walkers);
  }

  { // hamiltonian
    ScopedTimer ham_local(timers.hamiltonian_timer);

    std::vector<QMCHamiltonian::FullPrecRealType> new_energies(
        step_context.non_local_ops.getMoveKind() == TmoveKind::OFF
            ? ham_dispatcher.flex_evaluate(walker_hamiltonians, walker_twfs, walker_elecs)
            : ham_dispatcher.flex_evaluateWithToperator(walker_hamiltonians, walker_twfs, walker_elecs));

    auto resetSigNLocalEnergy = [](MCPWalker& walker, TrialWaveFunction& twf, auto local_energy, auto rr_acc,
                                   auto rr_prop) {
      walker.resetProperty(twf.getLogPsi(), twf.getPhase(), local_energy, rr_acc, rr_prop, 1.0);
    };

    for (int iw = 0; iw < walkers.size(); ++iw)
    {
      resetSigNLocalEnergy(walkers[iw], walker_twfs[iw], new_energies[iw], rr_accepted[iw], rr_proposed[iw]);
      FullPrecRealType branch_weight = sft.branch_engine.branchWeight(new_energies[iw], old_energies[iw]);
      walkers[iw].get().Weight *= branch_weight;
      if (rr_proposed[iw] > 0)
        walkers[iw].get().Age = 0;
      else
        walkers[iw].get().Age++;
    }
  }

  { // estimator collectables
    ScopedTimer collectable_local(timers.collectables_timer);

    // evaluate non-physical hamiltonian elements
    for (int iw = 0; iw < walkers.size(); ++iw)
      walker_hamiltonians[iw].auxHevaluate(walker_twfs[iw], walker_elecs[iw], walkers[iw]);

    // save properties into walker
    for (int iw = 0; iw < walkers.size(); ++iw)
      walker_hamiltonians[iw].saveProperty(walkers[iw].get().getPropertyBase());
  }

  if (accumulate_this_step)
  {
    ScopedTimer est_timer(timers.estimators_timer);
    crowd.accumulate(step_context.get_random_gen());
  }

  // collect walker logs
  crowd.collectStepWalkerLog(sft.global_step);

  { // T-moves
    ScopedTimer tmove_timer(dmc_timers.tmove_timer);

    const auto num_walkers = walkers.size();
    std::vector<int> walker_non_local_moves_accepted(num_walkers, 0);
    RefVector<MCPWalker> moved_nonlocal_walkers;
    RefVectorWithLeader<ParticleSet> moved_nonlocal_walker_elecs(crowd.get_walker_elecs()[0]);
    RefVectorWithLeader<TrialWaveFunction> moved_nonlocal_walker_twfs(crowd.get_walker_twfs()[0]);
    moved_nonlocal_walkers.reserve(num_walkers);
    moved_nonlocal_walker_elecs.reserve(num_walkers);
    moved_nonlocal_walker_twfs.reserve(num_walkers);

    walker_non_local_moves_accepted = ham_dispatcher.flex_makeNonLocalMoves(walker_hamiltonians, walker_twfs,
                                                                            walker_elecs, step_context.non_local_ops);

    for (int iw = 0; iw < walkers.size(); ++iw)
      if (walker_non_local_moves_accepted[iw] > 0)
      {
        crowd.incNonlocalAccept(walker_non_local_moves_accepted[iw]);
        moved_nonlocal_walkers.push_back(walkers[iw]);
        moved_nonlocal_walker_elecs.push_back(walker_elecs[iw]);
        moved_nonlocal_walker_twfs.push_back(walker_twfs[iw]);
      }

    if (moved_nonlocal_walkers.size())
    {
      twf_dispatcher.flex_evaluateGL(moved_nonlocal_walker_twfs, moved_nonlocal_walker_elecs, false);
      if (sft.qmcdrv_input.get_debug_checks() & DriverDebugChecks::CHECKGL_AFTER_TMOVE)
        checkLogAndGL(crowd, "checkGL_after_tmove", sft.serializing_crowd_walkers);
      ps_dispatcher.flex_saveWalker(moved_nonlocal_walker_elecs, moved_nonlocal_walkers);
    }
  }
}

template void DMCBatched::advanceWalkers<CoordsType::POS>(const StateForThread& sft,
                                                          Crowd& crowd,
                                                          DriverTimers& timers,
                                                          DMCTimers& dmc_timers,
                                                          DMCContextForSteps& step_context,
                                                          bool recompute,
                                                          bool accumulate_this_step);

template void DMCBatched::advanceWalkers<CoordsType::POS_SPIN>(const StateForThread& sft,
                                                               Crowd& crowd,
                                                               DriverTimers& timers,
                                                               DMCTimers& dmc_timers,
                                                               DMCContextForSteps& step_context,
                                                               bool recompute,
                                                               bool accumulate_this_step);

void DMCBatched::runDMCStep(int crowd_id,
                            const StateForThread& sft,
                            DriverTimers& timers,
                            DMCTimers& dmc_timers,
                            UPtrVector<DMCContextForSteps>& context_for_steps,
                            UPtrVector<Crowd>& crowds)
{
  Crowd& crowd = *(crowds[crowd_id]);

  if (crowd.size() == 0)
    return;

  auto& rng = context_for_steps[crowd_id]->get_random_gen();
  crowd.setRNGForHamiltonian(rng);

  const IndexType step = sft.step;
  // Are we entering the the last step of a block to recompute at?
  const bool recompute_this_step  = (sft.is_recomputing_block && (step + 1) == sft.steps_per_block);
  const bool accumulate_this_step = (step % sft.qmcdrv_input.get_estimator_measurement_period() == 0);
  const bool spin_move            = sft.population.get_golden_electrons().isSpinor();
  if (spin_move)
    advanceWalkers<CoordsType::POS_SPIN>(sft, crowd, timers, dmc_timers, *context_for_steps[crowd_id],
                                         recompute_this_step, accumulate_this_step);
  else
    advanceWalkers<CoordsType::POS>(sft, crowd, timers, dmc_timers, *context_for_steps[crowd_id], recompute_this_step,
                                    accumulate_this_step);
}

void DMCBatched::process(xmlNodePtr node)
{
  ScopedTimer local_timer(timers_.startup_timer);
  print_mem("DMCBatched before initialization", app_log());

  try
  {
    QMCDriverNew::AdjustedWalkerCounts awc =
        adjustGlobalWalkerCount(*myComm, walker_configs_ref_.getActiveWalkers(), qmcdriver_input_.get_total_walkers(),
                                qmcdriver_input_.get_walkers_per_rank(), dmcdriver_input_.get_reserve(),
                                determineNumCrowds(qmcdriver_input_.get_num_crowds(), rngs_.size()));

    steps_per_block_ =
        determineStepsPerBlock(awc.global_walkers, qmcdriver_input_.get_requested_samples(),
                               qmcdriver_input_.get_requested_steps(), qmcdriver_input_.get_max_blocks());

    initPopulationAndCrowds(awc);
    createStepContexts(crowds_.size());
  }
  catch (const UniformCommunicateError& ue)
  {
    myComm->barrier_and_abort(ue.what());
  }

  {
    ReportEngine PRE("DMC", "resetUpdateEngines");
    Timer init_timer;
    // Here DMC loads "Ensemble of cloned MCWalkerConfigurations"
    // I'd like to do away with this method in DMCBatched.

    app_log() << "  Creating the branching engine and walker controler" << std::endl;
    const auto refE_update_scheme = dmcdriver_input_.get_refenergy_update_scheme();
    app_log() << "    Reference energy is updated using the "
              << (refE_update_scheme == DMCRefEnergyScheme::UNLIMITED_HISTORY ? "unlimited_history" : "limited_history")
              << " scheme" << std::endl;
    branch_engine_ =
        std::make_unique<SFNBranch>(qmcdriver_input_.get_tau(), dmcdriver_input_.get_feedback(), refE_update_scheme);
    branch_engine_->put(node);

    walker_controller_ = std::make_unique<WalkerControl>(myComm, Random, dmcdriver_input_.get_reconfiguration());
    walker_controller_->setMinMax(population_.get_num_global_walkers(), 0);
    walker_controller_->start();
    walker_controller_->put(node);

    std::ostringstream o;
    if (dmcdriver_input_.get_reconfiguration())
      o << "  Fixed population using reconfiguration method\n";
    else
      o << "  Fluctuating population\n";

    o << "  Persistent walkers are killed after " << dmcdriver_input_.get_max_age() << " MC sweeps\n";
    o << "  BranchInterval = " << dmcdriver_input_.get_branch_interval() << "\n";
    o << "  Steps per block = " << steps_per_block_ << "\n";
    o << "  Number of blocks = " << qmcdriver_input_.get_max_blocks() << "\n";
    app_log() << o.str() << std::endl;

    app_log() << "  DMC Engine Initialization = " << init_timer.elapsed() << " secs" << std::endl;
  }

  if (qmcdriver_input_.get_measure_imbalance())
    measureImbalance("Startup");
}

void DMCBatched::run()
{
  IndexType num_blocks = qmcdriver_input_.get_max_blocks();

  estimator_manager_->startDriverRun();

  //initialize WalkerLogManager and collectors
  WalkerLogManager wlog_manager(walker_logs_input, allow_walker_logs, get_root_name(), myComm);
  for (auto& crowd : crowds_)
    crowd->setWalkerLogCollector(wlog_manager.makeCollector());
  //register walker log collectors into the manager
  wlog_manager.startRun(Crowd::getWalkerLogCollectorRefs(crowds_));

  StateForThread dmc_state(qmcdriver_input_, *drift_modifier_, *branch_engine_, population_, steps_per_block_,
                           serializing_crowd_walkers_);

  LoopTimer<> dmc_loop;
  RunTimeControl<> runtimeControl(run_time_manager, project_data_.getMaxCPUSeconds(), project_data_.getTitle(),
                                  myComm->rank() == 0);

  { // walker initialization
    ScopedTimer local_timer(timers_.init_walkers_timer);
    ParallelExecutor<> section_start_task;
    auto step_contexts_refs = getContextForStepsRefs();
    section_start_task(crowds_.size(), initialLogEvaluation, crowds_, step_contexts_refs, serializing_crowd_walkers_);

    FullPrecRealType energy, variance;
    population_.measureGlobalEnergyVariance(*myComm, energy, variance);
    // false indicates we do not support kill at node crossings.
    branch_engine_->initParam(population_, energy, variance, dmcdriver_input_.get_reconfiguration(), false);
    walker_controller_->setTrialEnergy(branch_engine_->getEtrial());

    print_mem("DMCBatched after initialLogEvaluation", app_summary());
    if (qmcdriver_input_.get_measure_imbalance())
      measureImbalance("InitialLogEvaluation");
  }

  // this barrier fences all previous load imbalance. Avoid block 0 timing pollution.
  myComm->barrier();

  ScopedTimer local_timer(timers_.production_timer);
  ParallelExecutor<> crowd_task;

  int global_step = 0;
  for (int block = 0; block < num_blocks; ++block)
  {
    {
      ScopeGuard<LoopTimer<>> dmc_local_timer(dmc_loop);
      estimator_manager_->startBlock(steps_per_block_);

      dmc_state.recalculate_properties_period = (qmc_driver_mode_[QMC_UPDATE_MODE])
          ? qmcdriver_input_.get_recalculate_properties_period()
          : (qmcdriver_input_.get_max_blocks() + 1) * steps_per_block_;
      dmc_state.is_recomputing_block          = qmcdriver_input_.get_blocks_between_recompute()
                   ? (1 + block) % qmcdriver_input_.get_blocks_between_recompute() == 0
                   : false;

      for (UPtr<Crowd>& crowd : crowds_)
        crowd->startBlock(steps_per_block_);

      for (int step = 0; step < steps_per_block_; ++step, ++global_step)
      {
        ScopedTimer local_timer(timers_.run_steps_timer);

        dmc_state.step        = step;
        dmc_state.global_step = global_step;
        crowd_task(crowds_.size(), runDMCStep, dmc_state, timers_, dmc_timers_, step_contexts_, crowds_);

        {
          const int iter = block * steps_per_block_ + step;
          walker_controller_->branch(iter, population_, iter == 0);
          branch_engine_->updateParamAfterPopControl(walker_controller_->get_ensemble_property(),
                                                     population_.get_golden_electrons().getTotalNum());
          walker_controller_->setTrialEnergy(branch_engine_->getEtrial());
        }

        population_.redistributeWalkers(crowds_);
      }
      print_mem("DMCBatched after a block", app_debug_stream());
      if (qmcdriver_input_.get_measure_imbalance())
        measureImbalance("Block " + std::to_string(block));
      endBlock();
      wlog_manager.writeBuffers();
      recordBlock(block);
    }

    bool stop_requested = false;
    // Rank 0 decides whether the time limit was reached
    if (!myComm->rank())
      stop_requested = runtimeControl.checkStop(dmc_loop);
    myComm->bcast(stop_requested);
    // Progress messages before possibly stopping
    if (!myComm->rank())
      app_log() << runtimeControl.generateProgressMessage("DMCBatched", block, num_blocks);
    if (stop_requested)
    {
      if (!myComm->rank())
        app_log() << runtimeControl.generateStopMessage("DMCBatched", block);
      run_time_manager.markStop();
      break;
    }
  }

  branch_engine_->printStatus();

  print_mem("DMCBatched ends", app_log());

  wlog_manager.stopRun();
  estimator_manager_->stopDriverRun();

  finalize(num_blocks, true);
}

RefVector<QMCDriverNew::ContextForSteps> DMCBatched::getContextForStepsRefs() const
{
  RefVector<ContextForSteps> refs;
  refs.reserve(step_contexts_.size());
  for (auto& one_context : step_contexts_)
    refs.push_back(*one_context);
  return refs;
}

void DMCBatched::createStepContexts(int num_crowds)
{
  assert(num_crowds <= rngs_.size());
  step_contexts_.resize(num_crowds);
  for (int i = 0; i < num_crowds; ++i)
    step_contexts_[i] =
        std::make_unique<DMCContextForSteps>(rngs_[i],
                                             NonLocalTOperator(population_.get_golden_hamiltonian().hasPhysicalNLPP()
                                                                   ? dmcdriver_input_.get_non_local_move()
                                                                   : TmoveKind::OFF,
                                                               qmcdriver_input_.get_tau(), dmcdriver_input_.get_alpha(),
                                                               dmcdriver_input_.get_gamma()));
}

} // namespace qmcplusplus
