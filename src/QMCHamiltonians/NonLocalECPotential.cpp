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


#include <cstdlib>
#include <iostream>
#include <atomic>
#include <mutex>

#include "NonLocalECPotential.h"
#include "OMPTarget/OffloadAlignedAllocators.hpp"

#include <optional>

#include <DistanceTable.h>
#include <IteratorUtility.h>
#include <ResourceCollection.h>
#include "NonLocalECPComponent.h"
#include "NonLocalTOperator.h"
#include "NLPPJob.h"

namespace qmcplusplus
{

struct NonLocalECPotential::NonLocalECPotentialMultiWalkerResource : public Resource
{
  NonLocalECPotentialMultiWalkerResource() : Resource("NonLocalECPotential") {}
  NonLocalECPotentialMultiWalkerResource(const NonLocalECPotentialMultiWalkerResource& other)
      : Resource("NonLocalECPotential"), collection(other.collection)
  {}

  std::unique_ptr<Resource> makeClone() const override
  { return std::make_unique<NonLocalECPotentialMultiWalkerResource>(*this); }

  ResourceCollection collection{"NLPPcollection"};
  /** scratch for building the neighbour list on the device.
   *
   * The electron-ion distances are computed on the device and were being copied back
   * in full every step so the host could apply a dist < Rmax cutoff. Only the pairs
   * that survive that cutoff are needed, and they are a small fraction, so the filter
   * runs on the device and only the survivors come back.
   */
  Vector<Real, OffloadPinnedAllocator<Real>> rmax_per_ion;
  Vector<int, OffloadPinnedAllocator<int>> job_counts;    // [nw]
  Vector<int, OffloadPinnedAllocator<int>> job_ion;       // [nw][job_stride]
  Vector<int, OffloadPinnedAllocator<int>> job_elec;
  Vector<Real, OffloadPinnedAllocator<Real>> job_dist;
  Vector<Real, OffloadPinnedAllocator<Real>> job_displ;   // 3 per job
  /// a crowds worth of per particle nonlocal ecp potential values
  Matrix<Real> ve_samples;
  Matrix<Real> vi_samples;
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
{ mw_evaluateImpl(o_list, wf_list, p_list, false, std::nullopt); }

NonLocalECPotential::Return_t NonLocalECPotential::evaluateWithToperator(TrialWaveFunction& psi, ParticleSet& P)
{
  evaluateImpl(psi, P, true);
  return value_;
}

void NonLocalECPotential::mw_evaluateWithToperator(const RefVectorWithLeader<OperatorBase>& o_list,
                                                   const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                                   const RefVectorWithLeader<ParticleSet>& p_list) const
{ mw_evaluateImpl(o_list, wf_list, p_list, true, std::nullopt); }

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
#if !defined(REMOVE_TRACEMANAGER)
          if (streaming_particles_)
          {
            Ve_samp(jel) += 0.5 * pairpot;
            Vi_samp(iat) += 0.5 * pairpot;
          }
#endif
        }
    }
  }

#if !defined(TRACE_CHECK) && !defined(REMOVE_TRACEMANAGER)
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

/** Build the per-walker neighbour lists on the device.
 *
 * The electron-ion distances are already there; the host scan that this replaces only
 * applies a dist < Rmax cutoff, and to do that the whole table is copied back every
 * step. Here the cutoff runs on the device and only the surviving pairs are returned,
 * which is a small fraction of the table.
 *
 * Layout of the multi-walker table, from SoaDistanceTableABOMPTarget: for global target
 * index t, distances start at t * stride_size with num_sources entries, and the
 * displacement components follow at offsets num_padded, 2 * num_padded and
 * 3 * num_padded within the same stride.
 *
 * @param materialize_jobs copy the compacted device output into each operator's job list
 * @return false when the device path is unavailable, leaving the caller on the host scan
 */
bool NonLocalECPotential::buildNeighborJobsOnDevice(const RefVectorWithLeader<OperatorBase>& o_list,
                                                    const RefVectorWithLeader<ParticleSet>& p_list,
                                                    int ig,
                                                    bool materialize_jobs)
{
  auto& O_leader              = o_list.getCastedLeader<NonLocalECPotential>();
  const ParticleSet& P_leader = p_list.getLeader();
  const auto& table           = P_leader.getDistTableAB(O_leader.myTableIndex);
  const Real* mw_dist         = nullptr;
  try
  {
    mw_dist = table.getMultiWalkerDataPtr();
  }
  catch (...)
  {
    return false; // table has no multi-walker device data; stay on the host scan
  }
  if (mw_dist == nullptr)
    return false;

  const Real* mw_dist_dev  = getOffloadDevicePtr(const_cast<Real*>(mw_dist));
  auto& res                = O_leader.mw_res_handle_.getResource();
  const size_t nw          = o_list.size();
  const size_t num_sources = O_leader.PP.size();
  const size_t num_padded  = getAlignedSize<Real>(num_sources);
  const size_t stride_size = num_padded * 4;
  const int first_elec     = P_leader.first(ig);
  const int last_elec      = P_leader.last(ig);
  const size_t nelec_group = last_elec - first_elec;
  const size_t nelec_total = P_leader.getTotalNum();

  // Rmax per ion, gathered once; a null component means the ion has no pseudopotential
  // and is excluded by a negative cutoff no distance can satisfy.
  if (res.rmax_per_ion.size() != num_sources)
  {
    res.rmax_per_ion.resize(num_sources);
    for (size_t iat = 0; iat < num_sources; iat++)
      res.rmax_per_ion[iat] = O_leader.PP[iat] ? static_cast<Real>(O_leader.PP[iat]->getRmax()) : Real(-1);
    res.rmax_per_ion.updateTo();
  }

  res.job_counts.resize(nw);
  size_t job_stride = nelec_group * 2 + 8;
  while (true)
  {
    res.job_ion.resize(nw * job_stride);
    res.job_elec.resize(nw * job_stride);
    res.job_dist.resize(nw * job_stride);
    res.job_displ.resize(nw * job_stride * 3);

    auto* counts_ptr = res.job_counts.data();
    auto* ion_ptr    = res.job_ion.data();
    auto* elec_ptr   = res.job_elec.data();
    auto* dist_ptr   = res.job_dist.data();
    auto* displ_ptr  = res.job_displ.data();
    auto* counts_dev = res.job_counts.device_data();
    auto* ion_dev    = res.job_ion.device_data();
    auto* elec_dev   = res.job_elec.device_data();
    auto* dist_dev   = res.job_dist.device_data();
    auto* displ_dev  = res.job_displ.device_data();
    auto* rmax_dev   = res.rmax_per_ion.device_data();

    PRAGMA_OFFLOAD("omp target teams distribute num_teams(nw) \
                    is_device_ptr(mw_dist_dev, counts_dev, ion_dev, elec_dev, dist_dev, displ_dev, rmax_dev)")
    for (size_t iw = 0; iw < nw; iw++)
    {
      int count = 0;
      for (int jel = first_elec; jel < last_elec; jel++)
      {
        const size_t t    = iw * nelec_total + jel;
        const Real* dists = mw_dist_dev + t * stride_size;
        for (size_t iat = 0; iat < num_sources; iat++)
          if (rmax_dev[iat] > Real(0) && dists[iat] < rmax_dev[iat])
          {
            const int job_id = count++;
            if (job_id >= static_cast<int>(job_stride))
              continue;

            const size_t slot = iw * job_stride + job_id;
            ion_dev[slot]     = static_cast<int>(iat);
            elec_dev[slot]    = jel;
            dist_dev[slot]    = dists[iat];
            // displacements are stored as the table's convention; the host scan negates
            displ_dev[slot * 3 + 0] = -dists[num_padded + iat];
            displ_dev[slot * 3 + 1] = -dists[2 * num_padded + iat];
            displ_dev[slot * 3 + 2] = -dists[3 * num_padded + iat];
          }
      }
      counts_dev[iw] = count;
    }

    PRAGMA_OFFLOAD("omp target update from(counts_ptr[:nw], ion_ptr[:nw*job_stride], elec_ptr[:nw*job_stride], \
                                           dist_ptr[:nw*job_stride], displ_ptr[:nw*job_stride*3])")
    size_t required_stride = 0;
    for (size_t iw = 0; iw < nw; ++iw)
      required_stride = std::max(required_stride, static_cast<size_t>(res.job_counts[iw]));
    if (required_stride <= job_stride)
      break;
    job_stride = required_stride;
  }

  if (materialize_jobs)
    for (size_t iw = 0; iw < nw; ++iw)
    {
      auto& O       = o_list.getCastedElement<NonLocalECPotential>(iw);
      auto& joblist = O.nlpp_jobs[ig];
      joblist.clear();
      joblist.reserve(res.job_counts[iw]);
      for (int job_id = 0; job_id < res.job_counts[iw]; ++job_id)
      {
        const size_t slot = iw * job_stride + job_id;
        joblist.emplace_back(res.job_ion[slot], res.job_elec[slot], res.job_dist[slot],
                             PosType(res.job_displ[slot * 3 + 0], res.job_displ[slot * 3 + 1],
                                     res.job_displ[slot * 3 + 2]));
      }
    }

  return true;
}

/** Tallies the device neighbour-job check across crowds.
 *
 * Every crowd runs the comparison concurrently, so printing per call interleaves the
 * lines and a torn line cannot be read as either a pass or a failure. The counts are
 * accumulated instead and reported once, which is also the only form in which
 * "every job matched" is a statement about the whole run rather than about whichever
 * lines survived intact.
 */
class NLPPJobCheck
{
public:
  static NLPPJobCheck& get()
  {
    static NLPPJobCheck singleton;
    return singleton;
  }

  void entered() { calls_.fetch_add(1, std::memory_order_relaxed); }
  void unavailable() { unavailable_.fetch_add(1, std::memory_order_relaxed); }

  void compared(size_t jobs, size_t bad)
  {
    jobs_.fetch_add(jobs, std::memory_order_relaxed);
    mismatches_.fetch_add(bad, std::memory_order_relaxed);
    groups_.fetch_add(1, std::memory_order_relaxed);
  }

  void countDisagreed(int ig, size_t host, int device)
  {
    disagreed_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex_);
    std::cerr << "NLPPCHECK count disagreement ig=" << ig << " host=" << host << " device=" << device << '\n';
  }

  ~NLPPJobCheck()
  {
    if (calls_.load() == 0)
      return;
    std::cerr << "NLPPCHECK summary: mw_evaluateImpl calls=" << calls_.load() << " group scans=" << groups_.load()
              << " jobs compared=" << jobs_.load() << " mismatches=" << mismatches_.load()
              << " count disagreements=" << disagreed_.load() << " device unavailable=" << unavailable_.load()
              << std::endl;
  }

private:
  NLPPJobCheck() = default;
  std::atomic<size_t> calls_{0}, groups_{0}, jobs_{0}, mismatches_{0}, disagreed_{0}, unavailable_{0};
  std::mutex mutex_;
};

void NonLocalECPotential::mw_evaluateImpl(const RefVectorWithLeader<OperatorBase>& o_list,
                                          const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                          const RefVectorWithLeader<ParticleSet>& p_list,
                                          bool compute_txy_all,
                                          const std::optional<ListenerOption<Real>> listeners,
                                          bool keep_grid)
{
  if (const char* c = std::getenv("QMCPACK_CHECK_DEVICE_NLPP_JOBS"); c && *c == '1')
    NLPPJobCheck::get().entered();
  auto& O_leader           = o_list.getCastedLeader<NonLocalECPotential>();
  ParticleSet& pset_leader = p_list.getLeader();
  const size_t nw          = o_list.size();

  // The scan further down reads the electron-ion table on the host: every electron
  // against every ion, for every walker, every step, to keep the pairs inside Rmax,
  // which are a small fraction of what it reads. buildNeighborJobsOnDevice applies the
  // same filter where the table already is and brings back only the survivors.
  //
  // This does not remove the table's own copy back to the host. That copy is what
  // J1OrbitalSoA reads through getDistRow and getDisplRow, so the table cannot carry
  // MW_EVALUATE_RESULT_NO_TRANSFER_TO_HOST while a one-body Jastrow shares it. What is
  // removed here is the O(nelec * nions) host traversal per walker per step.
  bool device_jobs = false;
  if (const char* c = std::getenv("QMCPACK_DEVICE_NLPP_JOBS"); c && *c == '1')
  {
    device_jobs = true;
    for (size_t iw = 0; iw < nw && device_jobs; iw++)
    {
      auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
      O.neighbor_lists.clear();
      for (int ig = 0; ig < pset_leader.groups(); ++ig)
        O.nlpp_jobs[ig].clear();
    }

    for (int ig = 0; ig < pset_leader.groups() && device_jobs; ++ig)
    {
      if (!buildNeighborJobsOnDevice(o_list, p_list, ig))
      {
        device_jobs = false; // no device table, every walker falls back together
        break;
      }
      for (size_t iw = 0; iw < nw; iw++)
      {
        auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
        for (const auto& job : O.nlpp_jobs[ig])
          O.neighbor_lists.addElecIonPair(job.electron_id, job.ion_id);
      }
    }

    if (!device_jobs) // a partial build must not leave half a list behind
      for (size_t iw = 0; iw < nw; iw++)
      {
        auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
        O.neighbor_lists.clear();
        for (int ig = 0; ig < pset_leader.groups(); ++ig)
          O.nlpp_jobs[ig].clear();
      }
  }

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

    const auto& myTable = P.getDistTableAB(O.myTableIndex);
    if (!device_jobs)
    {
      O.neighbor_lists.clear();
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
    }
    // NOTE: this scan reads the electron-ion table on the host, which is why the table
    // is copied back in full every step (SoaDistanceTableABOMPTarget, the
    // MW_EVALUATE_RESULT_NO_TRANSFER_TO_HOST branch). Only pairs inside Rmax matter and
    // they are a small fraction of the table, so the filter belongs on the device with
    // just the survivors returned. buildNeighborJobsOnDevice does that. While it is
    // being proven it runs alongside and its result is compared here rather than used,
    // because a neighbour list that silently disagrees would corrupt the energy.
    if (const char* check = std::getenv("QMCPACK_CHECK_DEVICE_NLPP_JOBS"); check && *check == '1' && iw == 0)
    {
      auto& res = O_leader.mw_res_handle_.getResource();
      for (int ig = 0; ig < P.groups(); ++ig)
        if (buildNeighborJobsOnDevice(o_list, p_list, ig, false))
        {
          const int dev_count = res.job_counts[0];
          // Scan the host table here rather than reading nlpp_jobs: under
          // QMCPACK_DEVICE_NLPP_JOBS that member is itself device-populated, and
          // comparing the device against a copy of itself always agrees.
          std::vector<NLPPJob<Real>> host_jobs;
          {
            const auto& myTable = p_list[0].getDistTableAB(O_leader.myTableIndex);
            for (int jel = p_list[0].first(ig); jel < p_list[0].last(ig); ++jel)
            {
              const auto& dist  = myTable.getDistRow(jel);
              const auto& displ = myTable.getDisplRow(jel);
              for (int iat = 0; iat < O_leader.PP.size(); iat++)
                if (O_leader.PP[iat] != nullptr && dist[iat] < O_leader.PP[iat]->getRmax())
                  host_jobs.emplace_back(iat, jel, dist[iat], -displ[iat]);
            }
          }
          if (static_cast<size_t>(dev_count) != host_jobs.size())
            NLPPJobCheck::get().countDisagreed(ig, host_jobs.size(), dev_count);
          else
          {
            size_t bad = 0;
            for (int j = 0; j < dev_count; j++)
              if (res.job_ion[j] != host_jobs[j].ion_id || res.job_elec[j] != host_jobs[j].electron_id ||
                  std::abs(res.job_dist[j] - host_jobs[j].ion_elec_dist) > Real(1e-6))
                bad++;
            NLPPJobCheck::get().compared(dev_count, bad);
          }
        }
        else
          NLPPJobCheck::get().unavailable();
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
        // The batch lists are compacted: a walker with no job at this jobid
        // is absent, so they can be shorter than nw. Index by batch slot,
        // matching the accumulation loop below.
        for (size_t j = 0; j < ecp_component_list.size(); j++)
          pairpots[j] =
              ecp_component_list[j].evaluateOne(pset_list[j], std::nullopt, batch_list[j].get().ion_id, psi_list[j],
                                                batch_list[j].get().electron_id, batch_list[j].get().ion_elec_dist,
                                                batch_list[j].get().ion_elec_displ,
                                                compute_txy_all ? makeOptionalRef<std::vector<NonLocalData>>(
                                                                      tmove_xy_all_batch_list[j])
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

std::vector<int> NonLocalECPotential::mw_makeNonLocalMovesPbyP(const RefVectorWithLeader<OperatorBase>& o_list,
                                                               const RefVectorWithLeader<TrialWaveFunction>& wf_list,
                                                               const RefVectorWithLeader<ParticleSet>& p_list,
                                                               NonLocalTOperator& move_op)
{
  const size_t nw = o_list.size();
  std::vector<int> num_accepted(nw, 0);

  if (move_op.getMoveKind() != TmoveKind::V1)
  {
    for (size_t iw = 0; iw < nw; iw++)
      num_accepted[iw] =
          o_list.getCastedElement<NonLocalECPotential>(iw).makeNonLocalMovesPbyP(wf_list[iw], p_list[iw], move_op);
    return num_accepted;
  }

  auto& O_leader           = o_list.getCastedLeader<NonLocalECPotential>();
  ParticleSet& pset_leader = p_list.getLeader();

  // per-walker candidate lists and single-electron job lists, rebuilt per electron
  std::vector<std::vector<NonLocalData>> tmove_xy(nw);
  std::vector<std::vector<NLPPJob<Real>>> jel_jobs(nw);

  auto pp_component = std::find_if(O_leader.PPset.begin(), O_leader.PPset.end(), [](auto& ptr) { return bool(ptr); });
  assert(pp_component != std::end(O_leader.PPset));

  // generate random numbers in the order exactly the same as serialization code path.
  // Note that: O.myRNG of the same batch are exactly identical and thus the order matters.
  // ad-hoc allocating rng_vals memory is sub-optimal and needs to be taken care.
  Matrix<RealType> rng_vals(nw, pset_leader.getTotalNum());
  for (int iw = 0; iw < rng_vals.rows(); iw++)
  {
    auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
    for (int jel = 0; jel < rng_vals.cols(); jel++)
      rng_vals[iw][jel] = (*O.myRNG)();
  }

  RefVectorWithLeader<NonLocalECPComponent> ecp_component_list(**pp_component);
  RefVectorWithLeader<ParticleSet> pset_list(pset_leader);
  RefVectorWithLeader<TrialWaveFunction> psi_list(wf_list.getLeader());
  RefVector<const NLPPJob<Real>> batch_list;
  RefVector<std::vector<NonLocalData>> tmove_xy_batch_list;
  RefVector<VirtualParticleSet> vp_list;
  std::vector<Real> pairpots(nw);

  ecp_component_list.reserve(nw);
  pset_list.reserve(nw);
  psi_list.reserve(nw);
  batch_list.reserve(nw);
  tmove_xy_batch_list.reserve(nw);

  for (int ig = 0; ig < pset_leader.groups(); ++ig) //loop over species
  {
    TrialWaveFunction::mw_prepareGroup(wf_list, p_list, ig);

    for (int jel = pset_leader.first(ig); jel < pset_leader.last(ig); ++jel)
    {
      // candidate ratio evaluations of this electron, batched across walkers.
      // Neighbor lists are the ones built at the preceding energy evaluation,
      // matching the single-walker sweep; distances are read fresh so accepts
      // of earlier electrons in this sweep are seen.
      size_t max_num_jobs = 0;
      for (size_t iw = 0; iw < nw; iw++)
      {
        auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
        const ParticleSet& P(p_list[iw]);
        auto& jobs = jel_jobs[iw];
        jobs.clear();
        tmove_xy[iw].clear();
        const auto& myTable = P.getDistTableAB(O.myTableIndex);
        const auto& dist    = myTable.getDistRow(jel);
        const auto& displ   = myTable.getDisplRow(jel);
        for (const int iat : O.neighbor_lists.getNeighboringIons(jel))
          jobs.emplace_back(iat, jel, dist[iat], -displ[iat]);
        max_num_jobs = std::max(max_num_jobs, jobs.size());
      }

      for (size_t jobid = 0; jobid < max_num_jobs; jobid++)
      {
        ecp_component_list.clear();
        pset_list.clear();
        psi_list.clear();
        batch_list.clear();
        tmove_xy_batch_list.clear();
        vp_list.clear();
        vp_list.reserve(nw);
        for (size_t iw = 0; iw < nw; iw++)
        {
          auto& O = o_list.getCastedElement<NonLocalECPotential>(iw);
          if (jobid < jel_jobs[iw].size())
          {
            const auto& job = jel_jobs[iw][jobid];
            ecp_component_list.push_back(*O.PP[job.ion_id]);
            pset_list.push_back(p_list[iw]);
            if (O.vp_)
              vp_list.push_back(*O.vp_);
            psi_list.push_back(wf_list[iw]);
            batch_list.push_back(job);
            tmove_xy_batch_list.push_back(tmove_xy[iw]);
          }
        }

        if (O_leader.vp_)
          NonLocalECPComponent::mw_evaluateOne(ecp_component_list, pset_list, {*O_leader.vp_, std::move(vp_list)},
                                               psi_list, batch_list, pairpots, tmove_xy_batch_list,
                                               O_leader.mw_res_handle_.getResource().collection, O_leader.use_DLA);
        else
          for (size_t j = 0; j < ecp_component_list.size(); j++)
            ecp_component_list[j].evaluateOne(pset_list[j], std::nullopt, batch_list[j].get().ion_id, psi_list[j],
                                              batch_list[j].get().electron_id, batch_list[j].get().ion_elec_dist,
                                              batch_list[j].get().ion_elec_displ,
                                              makeOptionalRef<std::vector<NonLocalData>>(tmove_xy_batch_list[j]),
                                              O_leader.use_DLA);
      }

      // selection and accepts per walker: identical calls, RNG draw order,
      // and accept bookkeeping as the single-walker v1 sweep
      for (size_t iw = 0; iw < nw; iw++)
      {
        auto& O                      = o_list.getCastedElement<NonLocalECPotential>(iw);
        const NonLocalData* oneTMove = move_op.selectMove(rng_vals[iw][jel], tmove_xy[iw]);
        if (oneTMove)
        {
          TrialWaveFunction& psi = wf_list[iw];
          ParticleSet& P         = p_list[iw];
          GradType grad_iat;
          if (P.makeMoveAndCheck(jel, oneTMove->Delta) && psi.calcRatioGrad(P, jel, grad_iat) != ValueType(0))
          {
            psi.acceptMove(P, jel, true);
            P.acceptMove(jel);
            num_accepted[iw]++;
          }
        }
      }
    }
  }

  for (size_t iw = 0; iw < nw; iw++)
    if (num_accepted[iw] > 0)
    {
      wf_list[iw].completeUpdates();
      // this step also updates electron positions on the device.
      p_list[iw].donePbyP(true);
    }

  return num_accepted;
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
{ return std::make_unique<NonLocalECPotential>(*this, qp); }

} // namespace qmcplusplus
