//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//                    Peter W. Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "NonLocalECPotential.h"

#include <algorithm>
#include <iterator>
#include <optional>

#include <DistanceTable.h>
#include <IteratorUtility.h>
#include <ResourceCollection.h>
#include "NonLocalECPComponent.h"
#include "NonLocalTOperator.h"
#include "NLPPJob.h"
#include "OhmmsPETE/OhmmsMatrix.h"

namespace qmcplusplus
{

struct NonLocalECPotential::NonLocalECPotentialMultiWalkerResource : public Resource
{
  NonLocalECPotentialMultiWalkerResource() : Resource("NonLocalECPotential") {}

  std::unique_ptr<Resource> makeClone() const override
  {
    return std::make_unique<NonLocalECPotentialMultiWalkerResource>(*this);
  }

  ResourceCollection collection{"NLPPcollection"};
  /// a crowds worth of per particle nonlocal ecp potential values
  Matrix<Real> ve_samples;
  Matrix<Real> vi_samples;
  /// T-move random values for one sweep, drawn in serial walker order
  Matrix<Real> tmove_rng_vals;
};

/** constructor
 *\param ions the positions of the ions
 *\param els the positions of the electrons
 *\param psi trial wavefunction
 */
NonLocalECPotential::NonLocalECPotential(ParticleSet& ions, ParticleSet& els, bool enable_DLA, bool use_VP)
    : ForceBase(ions, els),
      myRNG(nullptr),
      IonConfig(ions),
      use_DLA(enable_DLA),
      vp_(use_VP ? std::make_unique<VirtualParticleSet>(els) : nullptr),
      Peln(els),
      neighbor_lists(els.getTotalNum(), ions.getTotalNum(), PP)
{
  setEnergyDomain(POTENTIAL);
  twoBodyQuantumDomain(IonConfig, els);
  myTableIndex  = els.addTable(IonConfig);
  auto num_ions = IonConfig.getTotalNum();
  PP.resize(num_ions, nullptr);
  prefix_ = "FNL";
  PPset.resize(IonConfig.getSpeciesSet().getTotalNum());
  PulayTerm.resize(num_ions);
  update_mode_.set(NONLOCAL, 1);
  nlpp_jobs.resize(els.groups());
  for (size_t ig = 0; ig < els.groups(); ig++)
  {
    // this should be enough in most calculations assuming that every electron cannot be in more than two pseudo regions.
    nlpp_jobs[ig].reserve(2 * els.groupsize(ig));
  }
}

NonLocalECPotential::NonLocalECPotential(const NonLocalECPotential& nlpp, ParticleSet& els)
    : ForceBase(nlpp.IonConfig, els),
      myRNG(nullptr),
      IonConfig(nlpp.IonConfig),
      use_DLA(nlpp.use_DLA),
      vp_(nlpp.vp_ ? std::make_unique<VirtualParticleSet>(els, nlpp.vp_->getNumDistTables()) : nullptr),
      Peln(els),
      neighbor_lists(nlpp.neighbor_lists)
{
  setEnergyDomain(POTENTIAL);
  twoBodyQuantumDomain(IonConfig, els);
  myTableIndex  = els.addTable(IonConfig);
  auto num_ions = IonConfig.getTotalNum();
  PP.resize(num_ions, nullptr);
  prefix_ = "FNL";
  PPset.resize(IonConfig.getSpeciesSet().getTotalNum());
  PulayTerm.resize(num_ions);
  update_mode_.set(NONLOCAL, 1);
  nlpp_jobs.resize(els.groups());
  for (size_t ig = 0; ig < els.groups(); ig++)
  {
    // this should be enough in most calculations assuming that every electron cannot be in more than two pseudo regions.
    nlpp_jobs[ig].reserve(2 * els.groupsize(ig));
  }
  for (int ig = 0; ig < nlpp.PPset.size(); ++ig)
    if (nlpp.PPset[ig])
      addComponent(ig, std::make_unique<NonLocalECPComponent>(*nlpp.PPset[ig], els));
}

NonLocalECPotential::~NonLocalECPotential() = default;

#if !defined(REMOVE_TRACEMANAGER)
void NonLocalECPotential::contributeParticleQuantities() { request_.contribute_array(name_); }

void NonLocalECPotential::checkoutParticleQuantities(TraceManager& tm)
{
  streaming_particles_ = request_.streaming_array(name_);
  if (streaming_particles_)
  {
    Ve_sample = tm.checkout_real<1>(name_, Peln);
    Vi_sample = tm.checkout_real<1>(name_, IonConfig);
  }
}

void NonLocalECPotential::deleteParticleQuantities()
{
  if (streaming_particles_)
  {
    delete Ve_sample;
    delete Vi_sample;
  }
}
#endif

NonLocalECPotential::Return_t NonLocalECPotential::evaluate(TrialWaveFunction& psi, ParticleSet& P)
{
  evaluateImpl(psi, P, false);
  return value_;
}

NonLocalECPotential::Return_t NonLocalECPotential::evaluateDeterministic(TrialWaveFunction& psi, ParticleSet& P)
{
  evaluateImpl(psi, P, false, true);
  return value_;
}

void NonLocalECPotential::mw_evaluate(const RefVectorWithLeader<OperatorBase>& o_list,
                                      const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                      const RefVectorWithLeader<ParticleSet>& p_list) const
{
  mw_evaluateImpl(o_list, wf_list, p_list, false, std::nullopt);
}

NonLocalECPotential::Return_t NonLocalECPotential::evaluateWithToperator(TrialWaveFunction& psi, ParticleSet& P)
{
  evaluateImpl(psi, P, true);
  return value_;
}

void NonLocalECPotential::mw_evaluateWithToperator(const RefVectorWithLeader<OperatorBase>& o_list,
                                                   const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                                   const RefVectorWithLeader<ParticleSet>& p_list) const
{
  mw_evaluateImpl(o_list, wf_list, p_list, true, std::nullopt);
}

void NonLocalECPotential::mw_evaluatePerParticle(const RefVectorWithLeader<OperatorBase>& o_list,
                                                 const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                                 const RefVectorWithLeader<ParticleSet>& p_list,
                                                 const std::vector<ListenerVector<Real>>& listeners,
                                                 const std::vector<ListenerVector<Real>>& listeners_ions) const
{
  std::optional<ListenerOption<Real>> l_opt(std::in_place, listeners, listeners_ions);
  mw_evaluateImpl(o_list, wf_list, p_list, false, l_opt);
}

void NonLocalECPotential::mw_evaluatePerParticleWithToperator(
    const RefVectorWithLeader<OperatorBase>& o_list,
    const RefVectorWithLeader<TrialWaveFunction>& wf_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    const std::vector<ListenerVector<Real>>& listeners,
    const std::vector<ListenerVector<Real>>& listeners_ions) const
{
  std::optional<ListenerOption<Real>> l_opt(std::in_place, listeners, listeners_ions);
  mw_evaluateImpl(o_list, wf_list, p_list, true, l_opt);
}

void NonLocalECPotential::evaluateImpl(TrialWaveFunction& psi, ParticleSet& P, bool compute_txy_all, bool keep_grid)
{
  if (compute_txy_all)
    tmove_xy_all_.clear();

  value_ = 0.0;
#if !defined(REMOVE_TRACEMANAGER)
  auto& Ve_samp = *Ve_sample;
  auto& Vi_samp = *Vi_sample;
  if (streaming_particles_)
  {
    Ve_samp = 0.0;
    Vi_samp = 0.0;
  }
#endif

  if (!keep_grid)
    for (int ipp = 0; ipp < PPset.size(); ipp++)
      if (PPset[ipp])
        PPset[ipp]->rotateQuadratureGrid(generateRandomRotationMatrix(*myRNG));

  neighbor_lists.clear();
  const auto& myTable = P.getDistTableAB(myTableIndex);
  for (int ig = 0; ig < P.groups(); ++ig) //loop over species
  {
    psi.prepareGroup(P, ig);
    for (int jel = P.first(ig); jel < P.last(ig); ++jel)
    {
      const auto& dist  = myTable.getDistRow(jel);
      const auto& displ = myTable.getDisplRow(jel);
      for (int iat = 0; iat < PP.size(); iat++)
        if (PP[iat] != nullptr && dist[iat] < PP[iat]->getRmax())
        {
          Real pairpot =
              PP[iat]->evaluateOne(P, vp_ ? makeOptionalRef<VirtualParticleSet>(*vp_) : std::nullopt, iat, psi, jel,
                                   dist[iat], -displ[iat],
                                   compute_txy_all ? makeOptionalRef<std::vector<NonLocalData>>(tmove_xy_all_)
                                                   : std::nullopt,
                                   use_DLA);
          neighbor_lists.addElecIonPair(jel, iat);

          value_ += pairpot;
          if (streaming_particles_)
          {
            Ve_samp(jel) += 0.5 * pairpot;
            Vi_samp(iat) += 0.5 * pairpot;
          }
        }
    }
  }

#if !defined(TRACE_CHECK)
  if (streaming_particles_)
  {
    Return_t Vnow = value_;
    Real Visum    = Vi_sample->sum();
    Real Vesum    = Ve_sample->sum();
    Real Vsum     = Vesum + Visum;
    if (std::abs(Vsum - Vnow) > TraceManager::trace_tol)
    {
      app_log() << "accumtest: NonLocalECPotential::evaluate()" << std::endl;
      app_log() << "accumtest:   tot:" << Vnow << std::endl;
      app_log() << "accumtest:   sum:" << Vsum << std::endl;
      APP_ABORT("Trace check failed");
    }
    if (std::abs(Vesum - Visum) > TraceManager::trace_tol)
    {
      app_log() << "sharetest: NonLocalECPotential::evaluate()" << std::endl;
      app_log() << "sharetest:   e share:" << Vesum << std::endl;
      app_log() << "sharetest:   i share:" << Visum << std::endl;
      APP_ABORT("Trace check failed");
    }
  }
#endif
}

void NonLocalECPotential::mw_evaluateImpl(const RefVectorWithLeader<OperatorBase>& o_list,
                                          const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                          const RefVectorWithLeader<ParticleSet>& p_list,
                                          bool compute_txy_all,
                                          const std::optional<ListenerOption<Real>> listeners,
                                          bool keep_grid)
{
  auto& O_leader           = o_list.getCastedLeader<NonLocalECPotential>();
  ParticleSet& pset_leader = p_list.getLeader();
  const size_t nw          = o_list.size();

  for (size_t iw = 0; iw < nw; iw++)
  {
    auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
    const ParticleSet& P(p_list[iw]);

    if (compute_txy_all)
      O.tmove_xy_all_.clear();

    if (!keep_grid)
      for (int ipp = 0; ipp < O.PPset.size(); ipp++)
        if (O.PPset[ipp])
          O.PPset[ipp]->rotateQuadratureGrid(generateRandomRotationMatrix(*O.myRNG));

    O.neighbor_lists.clear();
    const auto& myTable = P.getDistTableAB(O.myTableIndex);
    for (int ig = 0; ig < P.groups(); ++ig) //loop over species
    {
      auto& joblist = O.nlpp_jobs[ig];
      joblist.clear();

      for (int jel = P.first(ig); jel < P.last(ig); ++jel)
      {
        const auto& dist  = myTable.getDistRow(jel);
        const auto& displ = myTable.getDisplRow(jel);
        for (int iat = 0; iat < O.PP.size(); iat++)
          if (O.PP[iat] != nullptr && dist[iat] < O.PP[iat]->getRmax())
          {
            O.neighbor_lists.addElecIonPair(jel, iat);
            joblist.emplace_back(iat, jel, dist[iat], -displ[iat]);
          }
      }
    }

    O.value_ = 0.0;
  }

  if (listeners)
  {
    auto& ve_samples = O_leader.mw_res_handle_.getResource().ve_samples;
    auto& vi_samples = O_leader.mw_res_handle_.getResource().vi_samples;
    ve_samples.resize(nw, pset_leader.getTotalNum());
    vi_samples.resize(nw, O_leader.IonConfig.getTotalNum());
  }

  // the VP of the first NonLocalECPComponent is responsible for holding the shared resource.
  auto pp_component = std::find_if(O_leader.PPset.begin(), O_leader.PPset.end(), [](auto& ptr) { return bool(ptr); });
  assert(pp_component != std::end(O_leader.PPset));

  RefVector<NonLocalECPotential> ecp_potential_list;
  RefVectorWithLeader<NonLocalECPComponent> ecp_component_list(**pp_component);
  RefVectorWithLeader<ParticleSet> pset_list(pset_leader);
  RefVectorWithLeader<TrialWaveFunction> psi_list(wf_list.getLeader());

  RefVector<const NLPPJob<Real>> batch_list;
  RefVector<VirtualParticleSet> vp_list;
  std::vector<Real> pairpots(nw);
  RefVector<std::vector<NonLocalData>> tmove_xy_all_batch_list;

  ecp_potential_list.reserve(nw);
  ecp_component_list.reserve(nw);
  pset_list.reserve(nw);
  psi_list.reserve(nw);
  batch_list.reserve(nw);
  tmove_xy_all_batch_list.reserve(nw);

  for (int ig = 0; ig < pset_leader.groups(); ++ig) //loop over species
  {
    TrialWaveFunction::mw_prepareGroup(wf_list, p_list, ig);

    // find the max number of jobs of all the walkers
    size_t max_num_jobs = 0;
    for (size_t iw = 0; iw < nw; iw++)
    {
      const auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
      max_num_jobs  = std::max(max_num_jobs, O.nlpp_jobs[ig].size());
    }

    for (size_t jobid = 0; jobid < max_num_jobs; jobid++)
    {
      ecp_potential_list.clear();
      ecp_component_list.clear();
      pset_list.clear();
      psi_list.clear();
      batch_list.clear();
      tmove_xy_all_batch_list.clear();
      vp_list.reserve(nw);
      for (size_t iw = 0; iw < nw; iw++)
      {
        auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
        if (jobid < O.nlpp_jobs[ig].size())
        {
          const auto& job = O.nlpp_jobs[ig][jobid];
          ecp_potential_list.push_back(O);
          ecp_component_list.push_back(*O.PP[job.ion_id]);
          pset_list.push_back(p_list[iw]);
          if (O.vp_)
            vp_list.push_back(*O.vp_);
          psi_list.push_back(wf_list[iw]);
          batch_list.push_back(job);
          if (compute_txy_all)
            tmove_xy_all_batch_list.push_back(O.tmove_xy_all_);
        }
      }

      if (O_leader.vp_)
        NonLocalECPComponent::mw_evaluateOne(ecp_component_list, pset_list, {*O_leader.vp_, std::move(vp_list)},
                                             psi_list, batch_list, pairpots, tmove_xy_all_batch_list,
                                             O_leader.mw_res_handle_.getResource().collection, O_leader.use_DLA);
      else
        for (size_t iw = 0; iw < nw; iw++)
          pairpots[iw] =
              ecp_component_list[iw].evaluateOne(pset_list[iw], std::nullopt, batch_list[iw].get().ion_id, psi_list[iw],
                                                 batch_list[iw].get().electron_id, batch_list[iw].get().ion_elec_dist,
                                                 batch_list[iw].get().ion_elec_displ,
                                                 compute_txy_all ? makeOptionalRef<std::vector<NonLocalData>>(
                                                                       tmove_xy_all_batch_list[iw])
                                                                 : std::nullopt,
                                                 O_leader.use_DLA);

      // Right now this is just over walker but could and probably should be over a set
      // larger than the walker count.  The easiest way to not complicate the per particle
      // reporting code would be to add the crowd walker index to the nlpp job meta data.
      for (size_t j = 0; j < ecp_potential_list.size(); j++)
      {
        NonLocalECPotential& ecp = ecp_potential_list[j];
        ecp.value_ += pairpots[j];

        if (listeners)
        {
          auto& ve_samples = O_leader.mw_res_handle_.getResource().ve_samples;
          auto& vi_samples = O_leader.mw_res_handle_.getResource().vi_samples;
          // CAUTION! This may not be so simple in the future
          int iw = j;
          ve_samples(iw, batch_list[j].get().electron_id) += 0.5 * pairpots[j];
          vi_samples(iw, batch_list[j].get().ion_id) += 0.5 * pairpots[j];
        }

#ifdef DEBUG_NLPP_BATCHED
        std::vector<NonLocalData> tmove_xy_dummy;
        Real check_value =
            ecp_component_list[j].evaluateOne(pset_list[j],
                                              ecp.vp_ ? makeOptionalRef<VirtualParticleSet>(*ecp.vp_) : std::nullopt,
                                              batch_list[j].get().ion_id, psi_list[j], batch_list[j].get().electron_id,
                                              batch_list[j].get().ion_elec_dist, batch_list[j].get().ion_elec_displ,
                                              compute_txy_all
                                                  ? makeOptionalRef<std::vector<NonLocalData>>(tmove_xy_dummy)
                                                  : std::nullopt,
                                              O_leader.use_DLA);
        if (std::abs(check_value - pairpots[j]) > 1e-5)
          std::cout << "check " << check_value << " wrong " << pairpots[j] << " diff "
                    << std::abs(check_value - pairpots[j]) << std::endl;
#endif
      }
    }
  }

  if (listeners)
  {
    // Motivation for this repeated definition is to make factoring this listener code out easy
    // and making it ignorable when reading this function.
    auto& ve_samples  = O_leader.mw_res_handle_.getResource().ve_samples;
    auto& vi_samples  = O_leader.mw_res_handle_.getResource().vi_samples;
    int num_electrons = pset_leader.getTotalNum();
    const std::string ion_listener_operator_name{O_leader.getName() + "Ion"};
    for (int iw = 0; iw < nw; ++iw)
    {
      Vector<Real> ve_sample(ve_samples.begin(iw), num_electrons);
      Vector<Real> vi_sample(vi_samples.begin(iw), O_leader.IonConfig.getTotalNum());
      for (const ListenerVector<Real>& listener : listeners->electron_values)
        listener.report(iw, O_leader.getName(), ve_sample);

      for (const ListenerVector<Real>& listener : listeners->ion_values)
        listener.report(iw, ion_listener_operator_name, vi_sample);
    }
    ve_samples = 0.0;
    vi_samples = 0.0;
  }
}

void NonLocalECPotential::evaluateIonDerivs(ParticleSet& P,
                                            ParticleSet& ions,
                                            TrialWaveFunction& psi,
                                            ParticleSet::ParticlePos& hf_terms,
                                            ParticleSet::ParticlePos& pulay_terms)
{
  value_    = 0.0;
  forces_   = 0;
  PulayTerm = 0;

  const auto& myTable = P.getDistTableAB(myTableIndex);
  for (int ig = 0; ig < P.groups(); ++ig) //loop over species
  {
    psi.prepareGroup(P, ig);
    for (int jel = P.first(ig); jel < P.last(ig); ++jel)
    {
      const auto& dist  = myTable.getDistRow(jel);
      const auto& displ = myTable.getDisplRow(jel);
      for (int iat = 0; iat < PP.size(); iat++)
        if (PP[iat] != nullptr && dist[iat] < PP[iat]->getRmax())
          value_ +=
              PP[iat]->evaluateOneWithForces(P, vp_ ? makeOptionalRef<VirtualParticleSet>(*vp_) : std::nullopt, ions,
                                             iat, psi, jel, dist[iat], -displ[iat], forces_[iat], PulayTerm);
    }
  }

  hf_terms -= forces_;
  pulay_terms -= PulayTerm;
}

void NonLocalECPotential::computeOneElectronTxy(TrialWaveFunction& psi,
                                                ParticleSet& P,
                                                const int ref_elec,
                                                std::vector<NonLocalData>& tmove_xy)
{
  tmove_xy.clear();
  const auto& myTable = P.getDistTableAB(myTableIndex);
  const auto& dist    = myTable.getDistRow(ref_elec);
  const auto& displ   = myTable.getDisplRow(ref_elec);
  for (const int iat : neighbor_lists.getNeighboringIons(ref_elec))
    PP[iat]->evaluateOne(P, vp_ ? makeOptionalRef<VirtualParticleSet>(*vp_) : std::nullopt, iat, psi, ref_elec,
                         dist[iat], -displ[iat], tmove_xy, use_DLA);
}

void NonLocalECPotential::evaluateOneBodyOpMatrix(ParticleSet& P,
                                                  const TWFFastDerivWrapper& psi,
                                                  std::vector<ValueMatrix>& B)
{
  bool keepGrid = true;
  for (int ipp = 0; ipp < PPset.size(); ipp++)
    if (PPset[ipp])
      if (!keepGrid)
        PPset[ipp]->rotateQuadratureGrid(generateRandomRotationMatrix(*myRNG));

  const auto& myTable = P.getDistTableAB(myTableIndex);
  for (int ig = 0; ig < P.groups(); ++ig) //loop over species
  {
    for (int jel = P.first(ig); jel < P.last(ig); ++jel)
    {
      const auto& dist  = myTable.getDistRow(jel);
      const auto& displ = myTable.getDisplRow(jel);
      for (int iat = 0; iat < PP.size(); iat++)
        if (PP[iat] != nullptr && dist[iat] < PP[iat]->getRmax())
          PP[iat]->evaluateOneBodyOpMatrixContribution(P, iat, psi, jel, dist[iat], -displ[iat], B);
    }
  }
}

void NonLocalECPotential::evaluateOneBodyOpMatrixForceDeriv(ParticleSet& P,
                                                            ParticleSet& source,
                                                            const TWFFastDerivWrapper& psi,
                                                            const int iat_source,
                                                            std::vector<std::vector<ValueMatrix>>& Bforce)
{
  bool keepGrid = true;
  for (int ipp = 0; ipp < PPset.size(); ipp++)
    if (PPset[ipp])
      if (!keepGrid)
        PPset[ipp]->rotateQuadratureGrid(generateRandomRotationMatrix(*myRNG));

  const auto& myTable = P.getDistTableAB(myTableIndex);
  for (int ig = 0; ig < P.groups(); ++ig) //loop over species
  {
    for (int jel = P.first(ig); jel < P.last(ig); ++jel)
    {
      const auto& dist  = myTable.getDistRow(jel);
      const auto& displ = myTable.getDisplRow(jel);
      for (int iat = 0; iat < PP.size(); iat++)
        if (PP[iat] != nullptr && dist[iat] < PP[iat]->getRmax())
          PP[iat]->evaluateOneBodyOpMatrixdRContribution(P, source, iat, iat_source, psi, jel, dist[iat], -displ[iat],
                                                         Bforce);
    }
  }
}

/** Electron-outer, walker-batched T-move sweep. See the header for why the electron loop
 *  stays serial and the walker loop does not.
 */
void NonLocalECPotential::mw_makeNonLocalMovesPbyP(const RefVectorWithLeader<OperatorBase>& o_list,
                                                   const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                                   const RefVectorWithLeader<ParticleSet>& p_list,
                                                   const RefVector<NonLocalTOperator>& move_op_list,
                                                   std::vector<int>& accepted_per_walker)
{
  auto& O_leader = o_list.getCastedLeader<NonLocalECPotential>();
  const size_t nw = o_list.size();
  accepted_per_walker.assign(nw, 0);
  if (nw == 0)
    return;

  const ParticleSet& P_leader = p_list.getLeader();

  // The batched Txy gather reads the multi-walker resource, and getResource() is
  // OptionalRef::value(): unheld it throws bad_optional_access from inside the crowd's
  // thread, which is how this aborted at crowds=8 while passing at 4. The sweep is an
  // optimisation, so an unheld handle falls back rather than failing.
  if (!O_leader.mw_res_handle_)
  {
    for (size_t iw = 0; iw < nw; iw++)
      accepted_per_walker[iw] =
          o_list.getCastedElement<NonLocalECPotential>(iw).makeNonLocalMovesPbyP(wf_list[iw], p_list[iw],
                                                                                move_op_list[iw]);
    return;
  }

  // Only the per-electron versions have an electron sweep to batch. V0 selects once from
  // the whole table and has a single move per step, so it goes down the serial path.
  const TmoveKind kind = move_op_list[0].get().getMoveKind();
  if (kind != TmoveKind::V1 && kind != TmoveKind::V3)
  {
    for (size_t iw = 0; iw < nw; iw++)
      accepted_per_walker[iw] =
          o_list.getCastedElement<NonLocalECPotential>(iw).makeNonLocalMovesPbyP(wf_list[iw], p_list[iw],
                                                                                 move_op_list[iw]);
    return;
  }

  for (size_t iw = 0; iw < nw; iw++)
  {
    auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
    if (kind == TmoveKind::V3)
    {
      O.elecTMAffected.assign(p_list[iw].getTotalNum(), false);
      move_op_list[iw].get().groupByElectron(p_list[iw].getTotalNum(), O.tmove_xy_all_);
    }
  }

  // Every walker in a crowd shares one generator: Crowd::setRNGForHamiltonian hands the
  // same rng to each walker's Hamiltonian. The serial sweep consumes it walker-major, one
  // draw per electron, so an electron-major sweep would assign different draws to different
  // (walker, electron) pairs and sample a different trajectory. Drawing the whole sweep up
  // front in serial order keeps the stream identical and lets the loops invert.
  auto& rng_vals = O_leader.mw_res_handle_.getResource().tmove_rng_vals;
  rng_vals.resize(nw, P_leader.getTotalNum());
  for (size_t iw = 0; iw < nw; iw++)
    for (size_t jel = 0; jel < P_leader.getTotalNum(); jel++)
      rng_vals[iw][jel] = (*o_list.getCastedElement<NonLocalECPotential>(iw).myRNG)();

  std::vector<std::vector<NonLocalData>> tmove_xy(nw);

  for (int ig = 0; ig < P_leader.groups(); ++ig)
  {
    TrialWaveFunction::mw_prepareGroup(wf_list, p_list, ig);

    for (int iat = P_leader.first(ig); iat < P_leader.last(ig); ++iat)
    {
      // Txy for this electron in every walker. The stale-table shortcut of V3 applies per
      // walker, so walkers that need a recomputation are gathered and the rest read their
      // grouped table; both end up with a tmove_xy to select from.
      // Txy for this electron across the walkers. computeOneElectronTxy is what calls
      // evaluateOne once per nearby ion, so leaving it in a per-walker loop is what keeps
      // the launch count at electrons times walkers; that is the cost this whole
      // restructure exists to remove. Walkers carry different numbers of nearby ions, so
      // the ion index is the outer loop and short walkers drop out of it, the same padding
      // mw_evaluateImpl uses with jobid.
      size_t max_ions = 0;
      for (size_t iw = 0; iw < nw; iw++)
      {
        auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
        tmove_xy[iw].clear();
        const bool fresh = (kind == TmoveKind::V1 || O.elecTMAffected[iat]);
        max_ions = std::max(max_ions, fresh ? O.neighbor_lists.getNeighboringIons(iat).size() : size_t(0));
      }

      // Which VirtualParticleSet actually holds the lent VPMultiWalkerMem, and which one
      // this sweep nominates as leader. mw_makeMoves reads mw_refPctls off the leader, so a
      // mismatch throws bad_optional_access from inside the crowd thread.
      if (const char* d = std::getenv("QMCPACK_DEBUG_TMOVE_LEADER"); d && *d == '1' && iat == P_leader.first(ig))
      {
        std::cerr << "TMLEAD ig=" << ig << " sweep_leader=" << static_cast<const void*>(O_leader.vp_.get())
                  << " held=" << (O_leader.vp_ ? O_leader.vp_->getMultiWalkerRefPctlsHeld() : false);
        for (size_t iw = 0; iw < nw; iw++)
        {
          auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
          std::cerr << " [iw=" << iw << " vp=" << static_cast<const void*>(O.vp_.get())
                    << " held=" << (O.vp_ ? O.vp_->getMultiWalkerRefPctlsHeld() : false) << "]";
        }
        std::cerr << std::endl;
      }

      if (O_leader.vp_ && max_ions > 0)
        for (size_t ionid = 0; ionid < max_ions; ionid++)
        {
          // Leaders stay the canonical walker-0 objects even when walker 0 contributes no
          // job, which is what mw_evaluateImpl does: it drops walkers from the contents with
          // jobid < size while keeping *O_leader.vp_ as leader. The resources were lent to
          // those objects, and every component's mw_evaluateRatios reads its handle off
          // getLeader(), so the leader must not move.
          RefVectorWithLeader<NonLocalECPComponent> comp_list(*O_leader.PPset[0]);
          RefVectorWithLeader<ParticleSet> tx_psets(p_list.getLeader());
          RefVectorWithLeader<TrialWaveFunction> tx_psis(wf_list.getLeader());
          RefVectorWithLeader<VirtualParticleSet> tx_vps(*O_leader.vp_);
          RefVector<const NLPPJob<RealType>> tx_jobs;
          RefVector<std::vector<NonLocalData>> tx_out;
          std::vector<NLPPJob<RealType>> job_store;
          job_store.reserve(nw);

          for (size_t iw = 0; iw < nw; iw++)
          {
            auto& O          = o_list.getCastedElement<NonLocalECPotential>(iw);
            const bool fresh = (kind == TmoveKind::V1 || O.elecTMAffected[iat]);
            if (!fresh)
              continue;
            const auto& ions = O.neighbor_lists.getNeighboringIons(iat);
            if (ionid >= ions.size())
              continue;
            const int ion   = ions[ionid];
            const auto& tbl = p_list[iw].getDistTableAB(O.myTableIndex);
            job_store.emplace_back(ion, iat, tbl.getDistRow(iat)[ion], -tbl.getDisplRow(iat)[ion]);
            comp_list.push_back(*O.PP[ion]);
            tx_psets.push_back(p_list[iw]);
            tx_psis.push_back(wf_list[iw]);
            tx_vps.push_back(*O.vp_);
            tx_out.push_back(tmove_xy[iw]);
          }
          if (job_store.empty())
            continue;

          // Does the gathered list contain the VP it nominates as leader? The team lock in
          // mw_evaluateOne acquires against vp_list.getLeader(), so a leader absent from the
          // list is lent a resource the call never uses, and the participating VPs are read
          // without one.
          if (const char* d = std::getenv("QMCPACK_DEBUG_TMOVE_LEADER"); d && *d == '1')
          {
            bool leader_present = false;
            for (const VirtualParticleSet& v : tx_vps)
              if (&v == O_leader.vp_.get())
                leader_present = true;
            std::cerr << "TMGATHER ig=" << ig << " iat=" << iat << " ionid=" << ionid
                      << " entries=" << tx_vps.size() << " leader=" << static_cast<const void*>(O_leader.vp_.get())
                      << " leader_in_list=" << leader_present
                      << " first=" << (tx_vps.size() ? static_cast<const void*>(&tx_vps[0]) : nullptr) << std::endl;
          }
          for (const auto& j : job_store)
            tx_jobs.push_back(j);

          std::vector<RealType> pairpots(job_store.size());
          NonLocalECPComponent::mw_evaluateOne(comp_list, tx_psets, {*O_leader.vp_, std::move(tx_vps)}, tx_psis,
                                               tx_jobs, pairpots, tx_out,
                                               O_leader.mw_res_handle_.getResource().collection, O_leader.use_DLA);
        }
      else
        for (size_t iw = 0; iw < nw; iw++)
        {
          auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
          if (kind == TmoveKind::V1 || O.elecTMAffected[iat])
            O.computeOneElectronTxy(wf_list[iw], p_list[iw], iat, tmove_xy[iw]);
        }

      // one selection per walker, then a single batched move and ratio for the walkers
      // that chose to move
      std::vector<int> movers;
      std::vector<ParticleSet::SingleParticlePos> deltas;
      movers.reserve(nw);
      deltas.reserve(nw);
      for (size_t iw = 0; iw < nw; iw++)
      {
        auto& O    = o_list.getCastedElement<NonLocalECPotential>(iw);
        auto& mop  = move_op_list[iw].get();
        const NonLocalData* one = (kind == TmoveKind::V1 || O.elecTMAffected[iat])
            ? mop.selectMove(rng_vals[iw][iat], tmove_xy[iw])
            : mop.selectMove(rng_vals[iw][iat], iat);
        if (one)
        {
          movers.push_back(static_cast<int>(iw));
          deltas.push_back(one->Delta);
        }
      }
      if (movers.empty())
        continue;

      // Leaders stay the canonical walker-0 objects. RealSpacePositionsOMPTarget::
      // mw_copyActivePos, reached through ParticleSet::mw_makeMove, reads its coordinate
      // resource off the list leader, and only walker 0's ParticleSet was lent one. Naming
      // movers[0], a walker picked by which T-moves were selected, throws
      // bad_optional_access from inside the crowd thread. gdb put the throw at
      // mw_copyActivePos; every earlier guess had it in mw_evaluateOne.
      // Two constraints pull in opposite directions. The list leader has to be walker 0,
      // which owns the lent resources every wavefunction component reads through
      // getLeader(); and mw_copyActivePos, reached from ParticleSet::mw_makeMove, writes
      // into leader-owned storage indexed by list POSITION, so the leader must also sit at
      // element 0. A list of movers alone satisfies neither when walker 0 has no move, which
      // NaNguard catches on CO2/Cu(110) as non-finite ratios in mw_calcRatioGrad; a list of
      // every walker satisfies both and costs a ratio evaluation per non-mover, which
      // measured 193 s against 78 s on diamondC.
      //
      // Walker 0 is therefore always element 0, with a zero displacement and rejection when
      // it is not moving, and only real movers follow it.
      // Walker 0 leads and sits at element 0; the movers follow. Both arrays are filled
      // in the same pass, since batch is built from movers and the correspondence is known
      // here: searching for it afterwards would be quadratic in a per-electron path.
      std::vector<int> batch;
      std::vector<ParticleSet::SingleParticlePos> batch_deltas;
      batch.reserve(movers.size() + 1);
      batch_deltas.reserve(movers.size() + 1);

      size_t zero_slot = movers.size();
      for (size_t m = 0; m < movers.size(); m++)
        if (movers[m] == 0)
          zero_slot = m;
      const bool zero_moves = zero_slot == movers.size();

      batch.push_back(0);
      batch_deltas.push_back(zero_moves ? ParticleSet::SingleParticlePos(0) : deltas[zero_slot]);
      for (size_t m = 0; m < movers.size(); m++)
        if (movers[m] != 0)
        {
          batch.push_back(movers[m]);
          batch_deltas.push_back(deltas[m]);
        }

      RefVectorWithLeader<ParticleSet> move_psets(p_list.getLeader());
      RefVectorWithLeader<TrialWaveFunction> move_twfs(wf_list.getLeader());
      for (int iw : batch)
      {
        move_psets.push_back(p_list[iw]);
        move_twfs.push_back(wf_list[iw]);
      }

      std::vector<bool> are_valid(batch.size(), true);
      ParticleSet::mw_makeMove(move_psets, iat, batch_deltas, makeOptionalRef(are_valid));
      std::vector<TrialWaveFunction::PsiValue> ratios(batch.size());
      TWFGrads<CoordsType::POS> grads(batch.size());
      TrialWaveFunction::mw_calcRatioGrad(move_twfs, move_psets, iat, ratios, grads);

      std::vector<bool> isAccepted(batch.size(), false);
      for (size_t b = 0; b < batch.size(); b++)
        isAccepted[b] = !(b == 0 && zero_moves) && are_valid[b] && ratios[b] != TrialWaveFunction::PsiValue(0);

      TrialWaveFunction::mw_accept_rejectMove(move_twfs, move_psets, iat, isAccepted, true);
      ParticleSet::mw_accept_rejectMove(move_psets, iat, isAccepted, true);

      for (size_t b = 0; b < batch.size(); b++)
        if (isAccepted[b])
        {
          const int iw = batch[b];
          auto& O      = o_list.getCastedElement<NonLocalECPotential>(iw);
          if (kind == TmoveKind::V3)
            O.neighbor_lists.markAffectedElecs(p_list[iw].getDistTableAB(O.myTableIndex), iat, O.elecTMAffected);
          accepted_per_walker[iw]++;
        }
    }
  }

  for (size_t iw = 0; iw < nw; iw++)
    if (accepted_per_walker[iw] > 0)
    {
      wf_list[iw].completeUpdates();
      p_list[iw].donePbyP(true);
    }
}

int NonLocalECPotential::makeNonLocalMovesPbyP(TrialWaveFunction& psi, ParticleSet& P, NonLocalTOperator& move_op)
{
  auto& RandomGen(*myRNG);
  auto& nonLocalOps = move_op;

  int NonLocalMoveAccepted = 0;
  if (move_op.getMoveKind() == TmoveKind::V0)
  {
    const NonLocalData* oneTMove = nonLocalOps.selectMove(RandomGen(), tmove_xy_all_);
    //make a non-local move
    if (oneTMove)
    {
      const int iat = oneTMove->PID;
      psi.prepareGroup(P, P.getGroupID(iat));
      GradType grad_iat;
      if (P.makeMoveAndCheck(iat, oneTMove->Delta) && psi.calcRatioGrad(P, iat, grad_iat) != ValueType(0))
      {
        psi.acceptMove(P, iat, true);
        P.acceptMove(iat);
        NonLocalMoveAccepted++;
      }
    }
  }
  else if (move_op.getMoveKind() == TmoveKind::V1)
  {
    GradType grad_iat;
    std::vector<NonLocalData> tmove_xy;
    //make a non-local move per particle
    for (int ig = 0; ig < P.groups(); ++ig) //loop over species
    {
      psi.prepareGroup(P, ig);
      for (int iat = P.first(ig); iat < P.last(ig); ++iat)
      {
        computeOneElectronTxy(psi, P, iat, tmove_xy);
        const NonLocalData* oneTMove = nonLocalOps.selectMove(RandomGen(), tmove_xy);
        if (oneTMove)
        {
          if (P.makeMoveAndCheck(iat, oneTMove->Delta) && psi.calcRatioGrad(P, iat, grad_iat) != ValueType(0))
          {
            psi.acceptMove(P, iat, true);
            P.acceptMove(iat);
            NonLocalMoveAccepted++;
          }
        }
      }
    }
  }
  else if (move_op.getMoveKind() == TmoveKind::V3)
  {
    elecTMAffected.assign(P.getTotalNum(), false);
    nonLocalOps.groupByElectron(P.getTotalNum(), tmove_xy_all_);
    GradType grad_iat;
    std::vector<NonLocalData> tmove_xy;
    //make a non-local move per particle
    for (int ig = 0; ig < P.groups(); ++ig) //loop over species
    {
      psi.prepareGroup(P, ig);
      for (int iat = P.first(ig); iat < P.last(ig); ++iat)
      {
        const NonLocalData* oneTMove;
        if (elecTMAffected[iat])
        {
          // recompute Txy for the given electron effected by T-moves
          computeOneElectronTxy(psi, P, iat, tmove_xy);
          oneTMove = nonLocalOps.selectMove(RandomGen(), tmove_xy);
        }
        else
          oneTMove = nonLocalOps.selectMove(RandomGen(), iat);
        if (oneTMove)
        {
          if (P.makeMoveAndCheck(iat, oneTMove->Delta) && psi.calcRatioGrad(P, iat, grad_iat) != ValueType(0))
          {
            psi.acceptMove(P, iat, true);
            // mark all affected electrons
            neighbor_lists.markAffectedElecs(P.getDistTableAB(myTableIndex), iat, elecTMAffected);
            P.acceptMove(iat);
            NonLocalMoveAccepted++;
          }
        }
      }
    }
  }

  if (NonLocalMoveAccepted > 0)
  {
    psi.completeUpdates();
    // this step also updates electron positions on the device.
    P.donePbyP(true);
  }

  return NonLocalMoveAccepted;
}

void NonLocalECPotential::addComponent(int groupID, std::unique_ptr<NonLocalECPComponent>&& ppot)
{
  for (int iat = 0; iat < PP.size(); iat++)
    if (IonConfig.GroupID[iat] == groupID)
      PP[iat] = ppot.get();
  PPset[groupID] = std::move(ppot);
}

void NonLocalECPotential::createResource(ResourceCollection& collection) const
{
  auto new_res = std::make_unique<NonLocalECPotentialMultiWalkerResource>();
  if (vp_)
    vp_->createResource(new_res->collection);
  auto resource_index = collection.addResource(std::move(new_res));
}

void NonLocalECPotential::acquireResource(ResourceCollection& collection,
                                          const RefVectorWithLeader<OperatorBase>& o_list) const
{
  auto& O_leader          = o_list.getCastedLeader<NonLocalECPotential>();
  O_leader.mw_res_handle_ = collection.lendResource<NonLocalECPotentialMultiWalkerResource>();
}

void NonLocalECPotential::releaseResource(ResourceCollection& collection,
                                          const RefVectorWithLeader<OperatorBase>& o_list) const
{
  auto& O_leader = o_list.getCastedLeader<NonLocalECPotential>();
  collection.takebackResource(O_leader.mw_res_handle_);
}

std::unique_ptr<OperatorBase> NonLocalECPotential::makeClone(ParticleSet& qp, TrialWaveFunction& psi) const
{
  return std::make_unique<NonLocalECPotential>(*this, qp);
}

} // namespace qmcplusplus
