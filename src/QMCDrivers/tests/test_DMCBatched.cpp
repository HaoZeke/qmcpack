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
#include "QMCDrivers/DMC/WalkerControl.h"
#include "QMCDrivers/tests/ValidQMCInputSections.h"
#include "QMCDrivers/tests/SetupDMCTest.h"
#include "EstimatorInputDelegates.h"
#include "Concurrency/Info.hpp"
#include "Concurrency/UtilityFunctions.hpp"
#include "Platforms/Host/OutputManager.h"
#include "SetupPools.h"

namespace qmcplusplus
{
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

TEST_CASE("DMCBatched::isPopulationControlStep", "[drivers]")
{
  // branch_interval = 1 controls every step, the current default behavior.
  for (int iter = 0; iter < 5; ++iter)
    CHECK(DMCBatched::isPopulationControlStep(iter, 1));

  // branch_interval = 3 controls on the first step and then at the end of
  // every three steps.
  CHECK(DMCBatched::isPopulationControlStep(0, 3));
  CHECK(!DMCBatched::isPopulationControlStep(1, 3));
  CHECK(DMCBatched::isPopulationControlStep(2, 3));
  CHECK(!DMCBatched::isPopulationControlStep(3, 3));
  CHECK(!DMCBatched::isPopulationControlStep(4, 3));
  CHECK(DMCBatched::isPopulationControlStep(5, 3));
}

TEST_CASE("WalkerControl::computeMultiplicity age damping", "[drivers]")
{
  using FPRT = WalkerControl::FullPrecRealType;
  constexpr FPRT rng_low{0.25};
  constexpr FPRT rng_high{0.75};

  // disabled policy (negative max_age): multiplicity is int(weight + rng) regardless of age
  CHECK(WalkerControl::computeMultiplicity(3.0, 100, -1, rng_low) == FPRT(3));
  CHECK(WalkerControl::computeMultiplicity(3.0, 0, -1, rng_high) == FPRT(3));

  // young walker (age 0): full weight branches
  CHECK(WalkerControl::computeMultiplicity(3.0, 0, 10, rng_low) == FPRT(3));

  // walker that failed to move this sweep: weight clamped to 1, cannot proliferate
  CHECK(WalkerControl::computeMultiplicity(3.0, 1, 10, rng_low) == FPRT(1));
  CHECK(WalkerControl::computeMultiplicity(3.0, 1, 10, rng_high) == FPRT(1));

  // persistent walker beyond max_age: weight clamped to 0.5, dies stochastically
  CHECK(WalkerControl::computeMultiplicity(3.0, 11, 10, rng_low) == FPRT(0));
  CHECK(WalkerControl::computeMultiplicity(3.0, 11, 10, rng_high) == FPRT(1));

  // clamps never raise a small weight
  CHECK(WalkerControl::computeMultiplicity(0.25, 11, 10, rng_low) == FPRT(0));
}

TEST_CASE("DMCDriverInput branchInterval parsing", "[drivers]")
{
  auto parse_interval = [](const std::string& body) {
    Libxml2Document doc;
    REQUIRE(doc.parseFromString("<qmc method=\"dmc\">" + body + "</qmc>"));
    DMCDriverInput input;
    input.readXML(doc.getRoot());
    return input.get_branch_interval();
  };

  CHECK(parse_interval("") == 1);
  CHECK(parse_interval("<parameter name=\"branchInterval\">3</parameter>") == 3);
  CHECK(parse_interval("<parameter name=\"branchinterval\">4</parameter>") == 4);
  CHECK(parse_interval("<parameter name=\"substeps\">5</parameter>") == 5);
  CHECK(parse_interval("<parameter name=\"subStep\">6</parameter>") == 6);
  // the historical "sub_stepd" typo alias is not accepted
  CHECK(parse_interval("<parameter name=\"sub_stepd\">7</parameter>") == 1);
  CHECK_THROWS(parse_interval("<parameter name=\"branchInterval\">0</parameter>"));
}

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
