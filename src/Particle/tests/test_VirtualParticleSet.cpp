//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:  Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Mark Dewing, markdewing@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include <MinimalParticlePool.h>
#include "Particle/VirtualParticleSet.h"
#include "ResourceCollection.h"
#include "DistanceTable.h"
#include "QMCHamiltonians/NLPPJob.h"

namespace qmcplusplus
{
TEST_CASE("VirtualParticleSet", "[particle]")
{
  auto pset_pool = MinimalParticlePool::make_NiO_a4(OHMMS::Controller);

  auto& ions  = *pset_pool.getParticleSet("i");
  auto& elecs = *pset_pool.getParticleSet("e");

  elecs.R[0] = {1, 2, 3};
  elecs.R[1] = {2, 1, 3};
  elecs.R[2] = {3, 1, 2};
  elecs.R[3] = {3, 2, 1};

  ions.addTable(ions);
  ions.update();
  elecs.addTable(ions);
  elecs.addTable(elecs);
  elecs.update();


  ParticleSet elecs_clone(elecs);
  elecs_clone.update();

  VirtualParticleSet vp_Ni(elecs);
  VirtualParticleSet vp_O(elecs);

  VirtualParticleSet vp_Ni_clone(elecs_clone);
  VirtualParticleSet vp_O_clone(elecs_clone);

  vp_Ni_clone.makeMoves(elecs_clone, 3, {{0.1, 0.2, 0.3}, {0.2, 0.1, 0.3}, {0.3, 0.1, 0.2}});
  const DistanceTableAB& dt_vp_ion = vp_Ni_clone.getDistTableAB(0);
  CHECK(Approx(dt_vp_ion.getDistances()[2][1]) == 2.5020600118);

  // two walkers form a workgroup.
  // One electron of the first walker gets inside O sphere and one electron of the other gets inside Ni sphere.
  RefVectorWithLeader<VirtualParticleSet> vp_list(vp_Ni, {vp_O, vp_Ni_clone});
  ResourceCollection collection{"NLPPcollection"};
  vp_Ni.createResource(collection);

  {
    ResourceCollectionTeamLock<VirtualParticleSet> vp_res_lock(collection, vp_list);
  }

  vp_Ni_clone.makeMoves(elecs_clone, 3, {{0.1, 0.2, 0.3}, {0.3, 0.1, 0.2}, {0.2, 0.1, 0.3}});
  CHECK(Approx(vp_Ni_clone.R[2][0]) == 3.2);
  CHECK(Approx(vp_Ni_clone.R[2][1]) == 2.1);
  CHECK(Approx(vp_Ni_clone.R[2][2]) == 1.3);

  REQUIRE(dt_vp_ion.getDistances().size() == 3);
  CHECK(Approx(dt_vp_ion.getDistances()[2][1]) == 2.5784519198);
}

/** A set spanning two ions of one electron.
 *
 * multi_source_ is assigned in exactly one place, inside mw_makeMovesMultiSource, and
 * nothing outside VirtualParticleSet calls that function. So isMultiSource() cannot
 * become true through any deck, and the three sites in HybridRepCenterOrbitals that
 * refuse a multi-source set are unreachable rather than merely uncovered. Reaching the
 * state here is what gives those refusals something to fire against.
 */
TEST_CASE("VirtualParticleSet multi-source", "[particle]")
{
  auto pset_pool = MinimalParticlePool::make_NiO_a4(OHMMS::Controller);

  auto& ions  = *pset_pool.getParticleSet("i");
  auto& elecs = *pset_pool.getParticleSet("e");

  elecs.R[0] = {1, 2, 3};
  elecs.R[1] = {2, 1, 3};
  elecs.R[2] = {3, 1, 2};
  elecs.R[3] = {3, 2, 1};

  ions.addTable(ions);
  ions.update();
  elecs.addTable(ions);
  elecs.addTable(elecs);
  elecs.update();

  ParticleSet elecs_two(elecs);
  elecs_two.update();

  VirtualParticleSet vp_one(elecs);
  VirtualParticleSet vp_two(elecs_two);

  RefVectorWithLeader<VirtualParticleSet> vp_list(vp_one, {vp_one, vp_two});
  RefVectorWithLeader<ParticleSet> refp_list(elecs, {elecs, elecs_two});

  using PosType  = VirtualParticleSet::PosType;
  using RealType = VirtualParticleSet::RealType;

  // walker 0 carries one ion of electron 1, walker 1 carries two ions of electron 2,
  // and the two ions get different quadrature offsets so a shared offset list would show
  std::vector<std::vector<NLPPJob<RealType>>> joblists(2);
  joblists[0].emplace_back(0, 1, RealType(1.5), PosType(0.1, 0.0, 0.0));
  joblists[1].emplace_back(0, 2, RealType(1.5), PosType(0.1, 0.0, 0.0));
  joblists[1].emplace_back(1, 2, RealType(2.0), PosType(0.0, 0.2, 0.0));

  std::vector<std::vector<std::vector<PosType>>> deltaV_lists(2);
  deltaV_lists[0] = {{{0.1, 0.2, 0.3}, {0.2, 0.1, 0.3}}};
  deltaV_lists[1] = {{{0.1, 0.2, 0.3}, {0.2, 0.1, 0.3}}, {{0.3, 0.1, 0.2}, {0.2, 0.3, 0.1}}};

  ResourceCollection collection{"NLPPcollection"};
  vp_one.createResource(collection);
  ResourceCollectionTeamLock<VirtualParticleSet> vp_res_lock(collection, vp_list);

  VirtualParticleSet::mw_makeMovesMultiSource(vp_list, refp_list, deltaV_lists, joblists, true);

  CHECK_FALSE(vp_one.isMultiSource());
  CHECK(vp_two.isMultiSource());

  // one job of two knots against two jobs of two knots
  CHECK(vp_one.getTotalNum() == 2);
  CHECK(vp_two.getTotalNum() == 4);

  // refSourcePtcl keeps the first job's ion, which is why reading it alone is wrong for
  // the two-ion walker, and the per virtual particle list carries what it loses
  CHECK(vp_two.refSourcePtcl == 0);
  REQUIRE(vp_two.source_ptcl_per_vp.size() == 4);
  CHECK(vp_two.source_ptcl_per_vp[0] == 0);
  CHECK(vp_two.source_ptcl_per_vp[1] == 0);
  CHECK(vp_two.source_ptcl_per_vp[2] == 1);
  CHECK(vp_two.source_ptcl_per_vp[3] == 1);

  // every job of a walker is the same electron
  CHECK(vp_two.refPtcl == 2);

  // which job each knot came from, so a consumer needing something per job rather than
  // per set or per knot can index by it
  REQUIRE(vp_two.job_per_vp.size() == 4);
  CHECK(vp_two.job_per_vp[0] == 0);
  CHECK(vp_two.job_per_vp[1] == 0);
  CHECK(vp_two.job_per_vp[2] == 1);
  CHECK(vp_two.job_per_vp[3] == 1);
  REQUIRE(vp_one.job_per_vp.size() == 2);
  CHECK(vp_one.job_per_vp[0] == 0);
  CHECK(vp_one.job_per_vp[1] == 0);

  // each job's own quadrature: the second job's knots are its own offsets from the same
  // electron, so a shared offset list would put R[2] at elecs_two.R[2] + {0.1,0.2,0.3}
  CHECK(Approx(vp_two.R[0][0]) == elecs_two.R[2][0] + 0.1);
  CHECK(Approx(vp_two.R[2][0]) == elecs_two.R[2][0] + 0.3);
  CHECK(Approx(vp_two.R[2][1]) == elecs_two.R[2][1] + 0.1);
  CHECK(Approx(vp_two.R[3][2]) == elecs_two.R[2][2] + 0.1);

  // both walkers here carry one electron each, whatever their ion count
  CHECK_FALSE(vp_one.isMultiRef());
  CHECK_FALSE(vp_two.isMultiRef());
}

/** A set whose jobs belong to different electrons.
 *
 * Collapsing the NLPP electron loop, which is where the offload launches are, puts every
 * job of a group into one set, and those jobs are different electrons. Each knot must be
 * placed against its own job's electron rather than the first job's, or the whole set is
 * silently offset.
 */
TEST_CASE("VirtualParticleSet multi-ref", "[particle]")
{
  auto pset_pool = MinimalParticlePool::make_NiO_a4(OHMMS::Controller);

  auto& ions  = *pset_pool.getParticleSet("i");
  auto& elecs = *pset_pool.getParticleSet("e");

  elecs.R[0] = {1, 2, 3};
  elecs.R[1] = {2, 1, 3};
  elecs.R[2] = {3, 1, 2};
  elecs.R[3] = {3, 2, 1};

  ions.addTable(ions);
  ions.update();
  elecs.addTable(ions);
  elecs.addTable(elecs);
  elecs.update();

  VirtualParticleSet vp(elecs);
  RefVectorWithLeader<VirtualParticleSet> vp_list(vp, {vp});
  RefVectorWithLeader<ParticleSet> refp_list(elecs, {elecs});

  using PosType  = VirtualParticleSet::PosType;
  using RealType = VirtualParticleSet::RealType;

  // one walker, two jobs, two different electrons
  std::vector<std::vector<NLPPJob<RealType>>> joblists(1);
  joblists[0].emplace_back(0, 1, RealType(1.5), PosType(0.1, 0.0, 0.0));
  joblists[0].emplace_back(0, 3, RealType(2.0), PosType(0.0, 0.2, 0.0));

  std::vector<std::vector<std::vector<PosType>>> deltaV_lists(1);
  deltaV_lists[0] = {{{0.1, 0.2, 0.3}}, {{0.4, 0.5, 0.6}}};

  ResourceCollection collection{"NLPPcollection"};
  vp.createResource(collection);
  ResourceCollectionTeamLock<VirtualParticleSet> vp_res_lock(collection, vp_list);

  VirtualParticleSet::mw_makeMovesMultiSource(vp_list, refp_list, deltaV_lists, joblists, true);

  CHECK(vp.isMultiRef());
  REQUIRE(vp.getTotalNum() == 2);

  // each knot sits on its own job's electron, 1 then 3, not on the first job's for both
  CHECK(Approx(vp.R[0][0]) == elecs.R[1][0] + 0.1);
  CHECK(Approx(vp.R[0][1]) == elecs.R[1][1] + 0.2);
  CHECK(Approx(vp.R[1][0]) == elecs.R[3][0] + 0.4);
  CHECK(Approx(vp.R[1][1]) == elecs.R[3][1] + 0.5);

  const VirtualParticleSet& const_vp = vp; // the public accessor is the const one
  const auto& refpctls               = const_vp.getMultiWalkerRefPctls();
  REQUIRE(refpctls.size() == 2);
  CHECK(refpctls[0] == 1);
  CHECK(refpctls[1] == 3);

  CHECK(vp.job_per_vp[0] == 0);
  CHECK(vp.job_per_vp[1] == 1);

  // and the electron per job, which is what selects an inverse row
  REQUIRE(vp.getNumJobs() == 2);
  CHECK(vp.job_electron[0] == 1);
  CHECK(vp.job_electron[1] == 3);

  /* The distance tables matter as much as the positions and are what a reader comparing
   * this function against mw_makeMoves cannot see. Build the same first job through
   * mw_makeMoves and require the table rows to agree: a set carrying two jobs must give
   * its first job's knots the same distances as a set carrying only that job.
   */
  const size_t ntables = vp.getNumDistTables();
  REQUIRE(ntables >= 1);
  std::vector<std::vector<double>> multi_rows(ntables);
  for (size_t t = 0; t < ntables; t++)
  {
    const auto& dt = vp.getDistTableAB(t);
    for (size_t k = 0; k < vp.getTotalNum(); k++)
      for (size_t s = 0; s < dt.sources(); s++)
        multi_rows[t].push_back(dt.getDistRow(k)[s]);
  }

  VirtualParticleSet vp_one_job(elecs);
  RefVectorWithLeader<VirtualParticleSet> one_list(vp_one_job, {vp_one_job});
  RefVectorWithLeader<ParticleSet> one_refp(elecs, {elecs});
  ResourceCollection one_collection{"NLPPcollection"};
  vp_one_job.createResource(one_collection);
  {
    ResourceCollectionTeamLock<VirtualParticleSet> lock(one_collection, one_list);
    const std::vector<PosType>& dv0 = deltaV_lists[0][0];
    RefVector<const std::vector<PosType>> dv_list{dv0};
    RefVector<const NLPPJob<RealType>> jl{joblists[0][0]};
    VirtualParticleSet::mw_makeMoves(one_list, one_refp, dv_list, jl, true);

    REQUIRE(vp_one_job.getTotalNum() == dv0.size());
    REQUIRE(vp_one_job.getNumDistTables() == ntables);
    for (size_t t = 0; t < ntables; t++)
    {
      const auto& dt1 = vp_one_job.getDistTableAB(t);
      for (size_t k = 0; k < vp_one_job.getTotalNum(); k++)
        for (size_t s = 0; s < dt1.sources(); s++)
          CHECK(Approx(dt1.getDistRow(k)[s]) == multi_rows[t][k * dt1.sources() + s]);
    }
  }
}

/** A set is reused between steps, so taking one reference has to retract the job map.
 *
 * multi_source_, multi_ref_, job_per_vp and job_electron describe the layout of a set,
 * and the orbital sets read them to decide whether to index inverse rows per job. A set
 * that once carried several jobs and then takes a single reference must stop advertising
 * them, or the next batch indexes through a map belonging to the previous one.
 */
TEST_CASE("VirtualParticleSet single reference retracts the job map", "[particle]")
{
  auto pset_pool = MinimalParticlePool::make_NiO_a4(OHMMS::Controller);

  auto& ions  = *pset_pool.getParticleSet("i");
  auto& elecs = *pset_pool.getParticleSet("e");

  elecs.R[0] = {1, 2, 3};
  elecs.R[1] = {2, 1, 3};
  elecs.R[2] = {3, 1, 2};
  elecs.R[3] = {3, 2, 1};

  ions.addTable(ions);
  ions.update();
  elecs.addTable(ions);
  elecs.addTable(elecs);
  elecs.update();

  VirtualParticleSet vp(elecs);
  RefVectorWithLeader<VirtualParticleSet> vp_list(vp, {vp});
  RefVectorWithLeader<ParticleSet> refp_list(elecs, {elecs});

  using PosType  = VirtualParticleSet::PosType;
  using RealType = VirtualParticleSet::RealType;

  ResourceCollection collection{"NLPPcollection"};
  vp.createResource(collection);
  ResourceCollectionTeamLock<VirtualParticleSet> vp_res_lock(collection, vp_list);

  // first a batch spanning two electrons, which sets the whole description
  std::vector<std::vector<NLPPJob<RealType>>> joblists(1);
  joblists[0].emplace_back(0, 1, RealType(1.5), PosType(0.1, 0.0, 0.0));
  joblists[0].emplace_back(2, 3, RealType(2.0), PosType(0.0, 0.2, 0.0));
  std::vector<std::vector<std::vector<PosType>>> deltaV_lists(1);
  deltaV_lists[0] = {{{0.1, 0.2, 0.3}, {0.2, 0.3, 0.4}}, {{0.4, 0.5, 0.6}}};

  VirtualParticleSet::mw_makeMovesMultiSource(vp_list, refp_list, deltaV_lists, joblists, true);

  REQUIRE(vp.isMultiRef());
  REQUIRE(vp.isMultiSource());
  REQUIRE(vp.getNumJobs() == 2);
  REQUIRE(vp.getTotalNum() == 3);

  // then a single job on the same set, through the ordinary path
  const std::vector<PosType> deltaV{{0.7, 0.8, 0.9}};
  const NLPPJob<RealType> job(1, 2, RealType(1.0), PosType(0.3, 0.0, 0.0));
  RefVector<const std::vector<PosType>> dv_list{deltaV};
  RefVector<const NLPPJob<RealType>> jl{job};
  VirtualParticleSet::mw_makeMoves(vp_list, refp_list, dv_list, jl, true);

  CHECK_FALSE(vp.isMultiRef());
  CHECK_FALSE(vp.isMultiSource());
  CHECK(vp.refPtcl == 2);
  CHECK(vp.refSourcePtcl == 1);

  // one job, naming this set's electron, and every knot mapped to it
  REQUIRE(vp.getNumJobs() == 1);
  CHECK(vp.job_electron[0] == 2);
  REQUIRE(vp.getTotalNum() == 1);
  REQUIRE(vp.job_per_vp.size() == vp.getTotalNum());
  CHECK(vp.job_per_vp[0] == 0);

  // the knot sits on the new electron, not on either of the previous batch's
  CHECK(Approx(vp.R[0][0]) == elecs.R[2][0] + 0.7);
  CHECK(Approx(vp.R[0][1]) == elecs.R[2][1] + 0.8);
  CHECK(Approx(vp.R[0][2]) == elecs.R[2][2] + 0.9);
}
} // namespace qmcplusplus
