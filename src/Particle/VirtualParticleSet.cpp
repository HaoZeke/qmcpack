//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/** @file VirtualParticleSet.cpp
 * A proxy class to the quantum ParticleSet
 */

#include "VirtualParticleSet.h"
#include <numeric>
#include "Configuration.h"
#include "Particle/DistanceTable.h"
#include "Particle/createDistanceTable.h"
#include "QMCHamiltonians/NLPPJob.h"
#include "ResourceCollection.h"

namespace qmcplusplus
{

struct VPMultiWalkerMem : public Resource
{
  /// multi walker reference particle
  Vector<int, OffloadPinnedAllocator<int>> mw_refPctls;

  VPMultiWalkerMem() : Resource("VPMultiWalkerMem") {}

  VPMultiWalkerMem(const VPMultiWalkerMem&) : VPMultiWalkerMem() {}

  std::unique_ptr<Resource> makeClone() const override { return std::make_unique<VPMultiWalkerMem>(*this); }
};

VirtualParticleSet::VirtualParticleSet(const ParticleSet& p, size_t dt_count_limit) : ParticleSet(p.getSimulationCell())
{
  setName("virtual");

  setSpinor(p.isSpinor());

  //create distancetables
  assert(dt_count_limit <= p.getNumDistTables());
  if (dt_count_limit == 0)
    dt_count_limit = p.getNumDistTables();

  std::ostream null_out(nullptr);
  for (int i = 0; i < dt_count_limit; ++i)
  {
    size_t tid = DistTables.size();
    auto& dt   = p.getDistTable(i);
    DistTables.push_back(createDistanceTable(dt.get_origin(), myName, null_out));
    if (!(dt.getModes() & DTModes::NEED_VP_FULL_TABLE_ON_HOST))
      DistTables[tid]->setModes(DTModes::MW_EVALUATE_RESULT_NO_TRANSFER_TO_HOST);
    app_debug() << "  ... VirtualParticleSet::VirtualParticleSet Create Table #" << tid << " "
                << DistTables[tid]->getName() << std::endl;
  }
}

void VirtualParticleSet::resize(const size_t nptcl)
{
  TotalNum = nptcl;
  R.resize(nptcl);
  if (isSpinor())
    spins.resize(nptcl);
  coordinates_->resize(nptcl);
}

VirtualParticleSet::~VirtualParticleSet() = default;

Vector<int, OffloadPinnedAllocator<int>>& VirtualParticleSet::getMultiWalkerRefPctls()
{
  return mw_mem_handle_.getResource().mw_refPctls;
}

const Vector<int, OffloadPinnedAllocator<int>>& VirtualParticleSet::getMultiWalkerRefPctls() const
{
  return mw_mem_handle_.getResource().mw_refPctls;
}

void VirtualParticleSet::createResource(ResourceCollection& collection) const
{
  collection.addResource(std::make_unique<VPMultiWalkerMem>());
  ParticleSet::createResource(collection);
}

void VirtualParticleSet::acquireResource(ResourceCollection& collection,
                                         const RefVectorWithLeader<VirtualParticleSet>& vp_list)
{
  auto& vp_leader          = vp_list.getLeader();
  vp_leader.mw_mem_handle_ = collection.lendResource<VPMultiWalkerMem>();

  auto p_list = RefVectorWithLeaderParticleSet(vp_list);
  ParticleSet::acquireResource(collection, p_list);
}

void VirtualParticleSet::releaseResource(ResourceCollection& collection,
                                         const RefVectorWithLeader<VirtualParticleSet>& vp_list)
{
  collection.takebackResource(vp_list.getLeader().mw_mem_handle_);
  auto p_list = RefVectorWithLeaderParticleSet(vp_list);
  ParticleSet::releaseResource(collection, p_list);
}


const RefVectorWithLeader<const DistanceTableAB> VirtualParticleSet::extractDTRefList(
    const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
    int id)
{
  RefVectorWithLeader<const DistanceTableAB> dt_list(vp_list.getLeader().getDistTableAB(id));
  dt_list.reserve(vp_list.size());
  for (const VirtualParticleSet& vp : vp_list)
  {
    const auto& d_table = vp.getDistTableAB(id);
    dt_list.push_back(d_table);
  }
  return dt_list;
}


const std::vector<QMCTraits::PosType> VirtualParticleSet::extractVPCoords(
    const RefVectorWithLeader<const VirtualParticleSet>& vp_list)
{
  std::vector<QMCTraits::PosType> coords_list;
  for (const VirtualParticleSet& vp : vp_list)
    for (int iat = 0; iat < vp.getTotalNum(); iat++)
      coords_list.push_back(vp.R[iat]);

  return coords_list;
}


/// move virtual particles to new postions and update distance tables
void VirtualParticleSet::makeMoves(const ParticleSet& refp,
                                   int jel,
                                   const std::vector<PosType>& deltaV,
                                   bool sphere,
                                   int iat)
{
  if (sphere && iat < 0)
    throw std::runtime_error(
        "VirtualParticleSet::makeMoves is invoked incorrectly, the flag sphere=true requires iat specified!");
  onSphere      = sphere;
  refPS         = refp;
  resize(deltaV.size());
  setSingleReference(jel, iat);
  for (size_t ivp = 0; ivp < R.size(); ivp++)
    R[ivp] = refp.R[jel] + deltaV[ivp];
  if (refp.isSpinor())
    for (size_t ivp = 0; ivp < R.size(); ivp++)
      spins[ivp] = refp.spins[jel]; //no spin deltas in this API
  update();
}

/// move virtual particles to new postions and update distance tables
void VirtualParticleSet::makeMovesWithSpin(const ParticleSet& refp,
                                           int jel,
                                           const std::vector<PosType>& deltaV,
                                           const std::vector<RealType>& deltaS,
                                           bool sphere,
                                           int iat)
{
  assert(refp.isSpinor());
  if (sphere && iat < 0)
    throw std::runtime_error(
        "VirtualParticleSet::makeMovesWithSpin is invoked incorrectly, the flag sphere=true requires iat specified!");
  onSphere      = sphere;
  refPS         = refp;
  resize(deltaV.size());
  setSingleReference(jel, iat);
  assert(deltaV.size() == deltaS.size());
  for (size_t ivp = 0; ivp < R.size(); ivp++)
  {
    R[ivp]     = refp.R[jel] + deltaV[ivp];
    spins[ivp] = refp.spins[jel] + deltaS[ivp];
  }
  update();
}

void VirtualParticleSet::setSingleReference(int electron_id, int ion_id)
{
  refPtcl       = electron_id;
  refSourcePtcl = ion_id;
  multi_source_ = false;
  multi_ref_    = false;
  job_electron.assign(1, electron_id);
  job_per_vp.assign(R.size(), 0);
  source_ptcl_per_vp.assign(R.size(), ion_id);
}

void VirtualParticleSet::mw_makeMovesMultiSource(const RefVectorWithLeader<VirtualParticleSet>& vp_list,
                                                const RefVectorWithLeader<ParticleSet>& refp_list,
                                                const std::vector<std::vector<std::vector<PosType>>>& deltaV_lists,
                                                const std::vector<std::vector<NLPPJob<RealType>>>& joblists,
                                                bool sphere)
{
  auto& vp_leader    = vp_list.getLeader();
  vp_leader.onSphere = sphere;
  vp_leader.refPS    = refp_list.getLeader();

  // Each job carries its own quadrature: deltaV depends on the job's ion-electron
  // displacement, so a walker's jobs cannot share one offset list.
  size_t nVPs = 0;
  for (size_t iw = 0; iw < vp_list.size(); iw++)
    for (size_t j = 0; j < joblists[iw].size(); j++)
      nVPs += deltaV_lists[iw][j].size();

  auto& mw_refPctls = vp_leader.getMultiWalkerRefPctls();
  mw_refPctls.resize(nVPs);
  RefVectorWithLeader<ParticleSet> p_list(vp_leader);
  p_list.reserve(vp_list.size());

  size_t ivp = 0;
  for (size_t iw = 0; iw < vp_list.size(); iw++)
  {
    VirtualParticleSet& vp(vp_list[iw]);
    const auto& deltaVs = deltaV_lists[iw];
    const auto& jobs    = joblists[iw];
    assert(deltaVs.size() == jobs.size());

    vp.onSphere      = sphere;
    vp.refPS         = refp_list[iw];
    vp.multi_source_ = jobs.size() > 1;
    // every job of a walker is the same electron, which is what refPtcl means; the source
    // varies and is kept per virtual particle
    vp.refPtcl       = jobs.empty() ? 0 : jobs[0].electron_id;
    vp.refSourcePtcl = jobs.empty() ? 0 : jobs[0].ion_id;
    size_t vp_count = 0;
    for (const auto& dv : deltaVs)
      vp_count += dv.size();
    vp.job_per_vp.resize(vp_count);
    vp.resize(vp_count);
    vp.source_ptcl_per_vp.resize(vp.R.size());

    size_t k = 0;
    vp.multi_ref_ = false;
    vp.job_electron.resize(jobs.size());
    for (size_t j = 0; j < jobs.size(); j++)
    {
      const auto& job    = jobs[j];
      const auto& deltaV = deltaVs[j];
      // Jobs of one set need not share an electron. Each knot is placed against its own
      // job's electron and records it, so a set spanning several is correct rather than
      // silently offset from the first job's position.
      if (job.electron_id != vp.refPtcl)
        vp.multi_ref_ = true;
      vp.job_electron[j] = job.electron_id;
      for (size_t q = 0; q < deltaV.size(); q++, k++, ivp++)
      {
        vp.R[k]                  = refp_list[iw].R[job.electron_id] + deltaV[q];
        vp.source_ptcl_per_vp[k] = job.ion_id;
        vp.job_per_vp[k]         = static_cast<int>(j);
        mw_refPctls[ivp]         = job.electron_id;
        if (vp_leader.isSpinor())
          vp.spins[k] = refp_list[iw].spins[job.electron_id];
      }
    }
    p_list.push_back(vp);
  }
  assert(ivp == nVPs);

  mw_refPctls.updateTo();
  ParticleSet::mw_update(p_list);
}

void VirtualParticleSet::mw_makeMoves(const RefVectorWithLeader<VirtualParticleSet>& vp_list,
                                      const RefVectorWithLeader<ParticleSet>& refp_list,
                                      const RefVector<const std::vector<PosType>>& deltaV_list,
                                      const RefVector<const NLPPJob<RealType>>& joblist,
                                      bool sphere)
{
  auto& vp_leader    = vp_list.getLeader();
  vp_leader.onSphere = sphere;
  vp_leader.refPS    = refp_list.getLeader();

  const size_t nVPs =
      std::accumulate(deltaV_list.begin(), deltaV_list.end(), 0,
                      [](size_t sum, const std::vector<PosType>& deltaV) { return sum + deltaV.size(); });
  auto& mw_refPctls = vp_leader.getMultiWalkerRefPctls();
  mw_refPctls.resize(nVPs);

  RefVectorWithLeader<ParticleSet> p_list(vp_leader);
  p_list.reserve(vp_list.size());

  size_t ivp = 0;
  for (int iw = 0; iw < vp_list.size(); iw++)
  {
    VirtualParticleSet& vp(vp_list[iw]);
    const std::vector<PosType>& deltaV(deltaV_list[iw]);
    const NLPPJob<RealType>& job(joblist[iw]);

    vp.onSphere      = sphere;
    vp.refPS         = refp_list[iw];
    vp.resize(deltaV.size());
    vp.setSingleReference(job.electron_id, job.ion_id);
    for (size_t k = 0; k < vp.R.size(); k++, ivp++)
    {
      vp.R[k] = refp_list[iw].R[vp.refPtcl] + deltaV[k];
      if (vp_leader.isSpinor())
        vp.spins[k] = refp_list[iw].spins[vp.refPtcl]; //no spin deltas in this API
      mw_refPctls[ivp] = vp.refPtcl;
    }
    p_list.push_back(vp);
  }
  assert(ivp == nVPs);

  mw_refPctls.updateTo();
  ParticleSet::mw_update(p_list);
}

void VirtualParticleSet::mw_makeMovesWithSpin(const RefVectorWithLeader<VirtualParticleSet>& vp_list,
                                              const RefVectorWithLeader<ParticleSet>& refp_list,
                                              const RefVector<const std::vector<PosType>>& deltaV_list,
                                              const RefVector<const std::vector<RealType>>& deltaS_list,
                                              const RefVector<const NLPPJob<RealType>>& joblist,
                                              bool sphere)
{
  auto& vp_leader = vp_list.getLeader();
  if (!vp_leader.isSpinor())
    throw std::runtime_error(
        "VirtualParticleSet::mw_makeMovesWithSpin should not be called if particle sets aren't spionor types");
  vp_leader.onSphere = sphere;
  vp_leader.refPS    = refp_list.getLeader();

  const size_t nVPs =
      std::accumulate(deltaV_list.begin(), deltaV_list.end(), 0,
                      [](size_t sum, const std::vector<PosType>& deltaV) { return sum + deltaV.size(); });
  auto& mw_refPctls = vp_leader.getMultiWalkerRefPctls();
  mw_refPctls.resize(nVPs);

  RefVectorWithLeader<ParticleSet> p_list(vp_leader);
  p_list.reserve(vp_list.size());

  size_t ivp = 0;
  for (int iw = 0; iw < vp_list.size(); iw++)
  {
    VirtualParticleSet& vp(vp_list[iw]);
    const std::vector<PosType>& deltaV(deltaV_list[iw]);
    const std::vector<RealType>& deltaS(deltaS_list[iw]);
    const NLPPJob<RealType>& job(joblist[iw]);

    vp.onSphere      = sphere;
    vp.refPS         = refp_list[iw];
    vp.resize(deltaV.size());
    vp.setSingleReference(job.electron_id, job.ion_id);
    assert(deltaV.size() == deltaS.size());
    for (size_t k = 0; k < vp.R.size(); k++, ivp++)
    {
      vp.R[k]          = refp_list[iw].R[vp.refPtcl] + deltaV[k];
      vp.spins[k]      = refp_list[iw].spins[vp.refPtcl] + deltaS[k];
      mw_refPctls[ivp] = vp.refPtcl;
    }
    p_list.push_back(vp);
  }
  assert(ivp == nVPs);

  mw_refPctls.updateTo();
  ParticleSet::mw_update(p_list);
}

} // namespace qmcplusplus
