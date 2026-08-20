//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "Message/Communicate.h"
#include "QMCDrivers/DMC/DMCDriverInput.h"
#include "QMCDrivers/DMC/DMCBatched.h"
#include "QMCDrivers/DMC/DMCDeviceAcceptance.h"
#include "QMCDrivers/tests/ValidQMCInputSections.h"
#include "QMCDrivers/tests/SetupDMCTest.h"
#include "EstimatorInputDelegates.h"
#include "Concurrency/Info.hpp"
#include "Concurrency/UtilityFunctions.hpp"
#include "Platforms/Host/OutputManager.h"
#include "SetupPools.h"

namespace qmcplusplus
{
TEST_CASE("DMC device acceptance predicate", "[drivers][dmc]")
{
  using RealType = QMCTraits::RealType;
  using PsiValue = QMCTraits::ValueType;

  constexpr size_t num_walkers = 6;
  Vector<PsiValue, OffloadPinnedAllocator<PsiValue>> ratios(num_walkers);
  Vector<RealType, OffloadPinnedAllocator<RealType>> log_gf(num_walkers);
  Vector<RealType, OffloadPinnedAllocator<RealType>> log_gb(num_walkers);
  Vector<RealType, OffloadPinnedAllocator<RealType>> variates(num_walkers);
  Vector<char, OffloadPinnedAllocator<char>> are_valid(num_walkers);
  Vector<char, OffloadPinnedAllocator<char>> accepted(num_walkers);

  const RealType epsilon = std::numeric_limits<RealType>::epsilon();
  ratios                 = {PsiValue(2), PsiValue(2), PsiValue(0), PsiValue(std::sqrt(epsilon / 2)),
                            PsiValue(0.5), PsiValue(0.5)};
  log_gf                  = RealType(0);
  log_gb                  = RealType(0);
  variates                = {RealType(0.5), RealType(0), RealType(0), RealType(0), RealType(0.25),
                             std::nextafter(RealType(0.25), RealType(0))};
  are_valid               = {1, 0, 1, 1, 1, 1};
  accepted                = char(-1);

  ratios.updateTo();
  log_gf.updateTo();
  log_gb.updateTo();
  variates.updateTo();
  are_valid.updateTo();
  accepted.updateTo();

  computeDMCDeviceAcceptance(num_walkers, ratios.device_data(), log_gf.device_data(), log_gb.device_data(),
                             are_valid.device_data(), variates.device_data(), accepted.device_data());
  accepted.updateFrom();

  CHECK(accepted[0] == 1);
  CHECK(accepted[1] == 0);
  CHECK(accepted[2] == 0);
  CHECK(accepted[3] == 0);
  CHECK(accepted[4] == 0);
  CHECK(accepted[5] == 1);
}

namespace testing
{
class DMCBatchedTest
{
public:
  DMCBatchedTest() { up_dtest_ = std::make_unique<SetupDMCTest>(1); }

private:
  UPtr<SetupDMCTest> up_dtest_;
};
} // namespace testing

/** Since we check the DMC only feature of reserve walkers perhaps this should be
 *  a DMC integration test.
 */
#ifdef _OPENMP
TEST_CASE("DMCDriver+QMCDriverNew integration", "[drivers]")
{
  using namespace testing;
  Concurrency::OverrideMaxCapacity<> override(8);
  RandomNumberGeneratorPool rng_pool(8);
  ProjectData test_project;
  Communicate* comm;
  comm = OHMMS::Controller;
  outputManager.pause();

  Libxml2Document doc;
  REQUIRE(doc.parseFromString(valid_dmc_input_sections[valid_dmc_input_dmc_batch_index]));
  xmlNodePtr node = doc.getRoot();
  QMCDriverInput qmcdriver_input;
  qmcdriver_input.readXML(node);
  DMCDriverInput dmcdriver_input;
  dmcdriver_input.readXML(node);
  auto particle_pool = MinimalParticlePool::make_diamondC_1x1x1(comm);
  auto wavefunction_pool =
      MinimalWaveFunctionPool::make_diamondC_1x1x1(test_project.getRuntimeOptions(), comm, particle_pool);

  auto hamiltonian_pool = MinimalHamiltonianPool::make_hamWithEE(comm, particle_pool, wavefunction_pool);
  SampleStack samples;
  WalkerConfigurations walker_confs;

  DMCBatched dmcdriver(test_project, std::move(qmcdriver_input), nullptr, std::move(dmcdriver_input), walker_confs,
                       MCPopulation(comm->size(), comm->rank(), *particle_pool.getParticleSet("e"),
                                    wavefunction_pool.getWaveFunction().value(),
                                    hamiltonian_pool.getHamiltonian().value()),
                       rng_pool.getRngRefs(), comm);

  // setStatus must be called before process
  std::string root_name{"Test"};
  //For later sections this appears to contain important state.
  std::string prev_config_file{""};

  dmcdriver.setStatus(root_name, prev_config_file, false);
  // We want to express out expectations of the QMCDriver state machine so we catch
  // changes to it over time.
  outputManager.resume();

  dmcdriver.process(node);
  CHECK(dmcdriver.get_num_living_walkers() == 8);
  const QMCTraits::IndexType reserved_walkers = dmcdriver.get_num_living_walkers() + dmcdriver.get_num_dead_walkers();
  CHECK(reserved_walkers == 10);
  // What else should we expect after process
}
#endif

} // namespace qmcplusplus
