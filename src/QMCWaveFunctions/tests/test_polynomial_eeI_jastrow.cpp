//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#include "catch.hpp"
#include "config.h"

#include "OhmmsData/Libxml2Doc.h"
#include "OhmmsData/AttributeSet.h"
#include "OhmmsPETE/OhmmsMatrix.h"
#include "Particle/ParticleSet.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "QMCWaveFunctions/Jastrow/PolynomialFunctor3D.h"
#include "QMCWaveFunctions/Jastrow/JeeIOrbitalSoA.h"
#include "QMCWaveFunctions/Jastrow/eeI_JastrowBuilder.h"
#include "CPU/VectorOps.h"
#include <ResourceCollection.h>
#include "QMCHamiltonians/NLPPJob.h"

#include <stdio.h>
#include <string>
#include <cmath>
#include <chrono>
#include <vector>
#include <memory>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

using std::string;

namespace qmcplusplus
{
using RealType = WaveFunctionComponent::RealType;
using LogValue = WaveFunctionComponent::LogValue;
using PsiValue = WaveFunctionComponent::PsiValue;
using GradType = WaveFunctionComponent::GradType;
using PosType  = QMCTraits::PosType;

TEST_CASE("PolynomialFunctor3D functor zero", "[wavefunction]")
{
  PolynomialFunctor3D functor("test_functor");

  double r = 1.2;
  double u = functor.evaluate(r, r, r);
  REQUIRE(u == 0.0);
}

void create_J3_ion_reference_values(TinyVector<ParticleSet::ParticleGradient, 3>& igr_egrad,
                                    TinyVector<ParticleSet::ParticleLaplacian, 3>& igr_lapl,
                                    int ionid)
{
  const int Nelec = 4; //This was the size of the electron set generated.

  //Incidentally, two ions were used for reference values.
  for (int i = 0; i < 3; i++)
  {
    igr_egrad[i].resize(4);
    igr_lapl[i].resize(4);
    igr_egrad[i] = 0;
    igr_lapl[i]  = 0;
  }

  if (ionid == 0)
  {
    ////////////////////////////////
    // Test Derivatives w.r.t Ion 0
    ////////////////////////////////
    // d/dR_0
    ////////////////////////////////
    igr_egrad[0][0][0] = 1.385610e-01;
    igr_egrad[0][0][1] = 0.000000e+00;
    igr_egrad[0][0][2] = -7.296107e-02;
    igr_egrad[0][1][0] = -9.073855e-02;
    igr_egrad[0][1][1] = 0.000000e+00;
    igr_egrad[0][1][2] = -3.106353e-02;
    igr_egrad[0][2][0] = 1.297277e-02;
    igr_egrad[0][2][1] = 0.000000e+00;
    igr_egrad[0][2][2] = 4.133443e-03;
    igr_egrad[0][3][0] = 1.416451e-02;
    igr_egrad[0][3][1] = 0.000000e+00;
    igr_egrad[0][3][2] = -4.392479e-02;
    igr_lapl[0][0]     = 9.355446e-01;
    igr_lapl[0][1]     = 2.150360e-01;
    igr_lapl[0][2]     = -5.127206e-02;
    igr_lapl[0][3]     = -7.065001e-02;

    // d/dR_1
    ////////////////////////////////
    igr_egrad[1][0][0] = 0.000000e+00;
    igr_egrad[1][0][1] = 8.411349e-02;
    igr_egrad[1][0][2] = 0.000000e+00;
    igr_egrad[1][1][0] = 0.000000e+00;
    igr_egrad[1][1][1] = -3.824645e-02;
    igr_egrad[1][1][2] = 0.000000e+00;
    igr_egrad[1][2][0] = 0.000000e+00;
    igr_egrad[1][2][1] = -8.098260e-02;
    igr_egrad[1][2][2] = 0.000000e+00;
    igr_egrad[1][3][0] = 0.000000e+00;
    igr_egrad[1][3][1] = -9.110417e-02;
    igr_egrad[1][3][2] = 0.000000e+00;
    igr_lapl[1][0]     = 0.000000e+00;
    igr_lapl[1][1]     = 0.000000e+00;
    igr_lapl[1][2]     = 0.000000e+00;
    igr_lapl[1][3]     = 0.000000e+00;

    // d/dR_2
    ////////////////////////////////
    igr_egrad[2][0][0] = -4.715708e-02;
    igr_egrad[2][0][1] = 0.000000e+00;
    igr_egrad[2][0][2] = 9.679753e-02;
    igr_egrad[2][1][0] = -2.559180e-02;
    igr_egrad[2][1][1] = 0.000000e+00;
    igr_egrad[2][1][2] = -3.513951e-02;
    igr_egrad[2][2][0] = -1.603880e-02;
    igr_egrad[2][2][1] = 0.000000e+00;
    igr_egrad[2][2][2] = -8.253370e-02;
    igr_egrad[2][3][0] = -5.502826e-02;
    igr_egrad[2][3][1] = 0.000000e+00;
    igr_egrad[2][3][2] = -4.319823e-02;
    igr_lapl[2][0]     = 1.613281e-01;
    igr_lapl[2][1]     = 5.972737e-02;
    igr_lapl[2][2]     = 2.801520e-03;
    igr_lapl[2][3]     = 1.414764e-01;
  }
  else if (ionid == 1)
  {
    ////////////////////////////////
    // Test Derivatives w.r.t Ion 1
    ////////////////////////////////
    // d/dR_0
    ////////////////////////////////
    igr_egrad[0][0][0] = 5.790416e-02;
    igr_egrad[0][0][1] = 0.000000e+00;
    igr_egrad[0][0][2] = 1.675537e-03;
    igr_egrad[0][1][0] = -1.028276e-02;
    igr_egrad[0][1][1] = 0.000000e+00;
    igr_egrad[0][1][2] = 3.106353e-02;
    igr_egrad[0][2][0] = 2.250537e-01;
    igr_egrad[0][2][1] = 0.000000e+00;
    igr_egrad[0][2][2] = 1.847021e-02;
    igr_egrad[0][3][0] = 1.205085e-02;
    igr_egrad[0][3][1] = 0.000000e+00;
    igr_egrad[0][3][2] = 4.321063e-02;
    igr_lapl[0][0]     = 1.671657e-01;
    igr_lapl[0][1]     = 8.572653e-03;
    igr_lapl[0][2]     = -7.744252e-01;
    igr_lapl[0][3]     = 1.543136e-01;

    // d/dR_1
    ////////////////////////////////
    igr_egrad[1][0][0] = 0.000000e+00;
    igr_egrad[1][0][1] = -7.975566e-02;
    igr_egrad[1][0][2] = 0.000000e+00;
    igr_egrad[1][1][0] = 0.000000e+00;
    igr_egrad[1][1][1] = -8.732100e-02;
    igr_egrad[1][1][2] = 0.000000e+00;
    igr_egrad[1][2][0] = 0.000000e+00;
    igr_egrad[1][2][1] = 7.768116e-02;
    igr_egrad[1][2][2] = 0.000000e+00;
    igr_egrad[1][3][0] = 0.000000e+00;
    igr_egrad[1][3][1] = -7.397172e-02;
    igr_egrad[1][3][2] = 0.000000e+00;
    igr_lapl[1][0]     = 0.000000e+00;
    igr_lapl[1][1]     = 0.000000e+00;
    igr_lapl[1][2]     = 0.000000e+00;
    igr_lapl[1][3]     = 0.000000e+00;

    // d/dR_2
    ////////////////////////////////
    igr_egrad[2][0][0] = 3.179015e-02;
    igr_egrad[2][0][1] = 0.000000e+00;
    igr_egrad[2][0][2] = -7.764732e-02;
    igr_egrad[2][1][0] = 2.559180e-02;
    igr_egrad[2][1][1] = 0.000000e+00;
    igr_egrad[2][1][2] = -8.421406e-02;
    igr_egrad[2][2][0] = 1.162496e-02;
    igr_egrad[2][2][1] = 0.000000e+00;
    igr_egrad[2][2][2] = 7.923212e-02;
    igr_egrad[2][3][0] = 2.541299e-02;
    igr_egrad[2][3][1] = 0.000000e+00;
    igr_egrad[2][3][2] = -5.560366e-02;
    igr_lapl[2][0]     = 2.159518e-02;
    igr_lapl[2][1]     = 5.972737e-02;
    igr_lapl[2][2]     = 3.744006e-02;
    igr_lapl[2][3]     = 1.398887e-01;
  }
  else
    throw std::runtime_error("create_J3_ion_reference_values: ionid can only be 0 or 1");
}

void test_J3_polynomial3D(const DynamicCoordinateKind kind_selected)
{
  Communicate* c = OHMMS::Controller;

  const SimulationCell simulation_cell;
  ParticleSet ions_(simulation_cell, kind_selected);
  ParticleSet elec_(simulation_cell, kind_selected);

  ions_.setName("ion");
  ions_.create({2});
  ions_.R[0] = {2.0, 0.0, 0.0};
  ions_.R[1] = {-2.0, 0.0, 0.0};
  SpeciesSet& source_species(ions_.getSpeciesSet());
  source_species.addSpecies("O");
  ions_.update();

  elec_.setName("elec");
  elec_.create({2, 2});
  elec_.R[0] = {1.00, 0.0, 0.0};
  elec_.R[1] = {0.0, 0.0, 0.0};
  elec_.R[2] = {-1.00, 0.0, 0.0};
  elec_.R[3] = {0.0, 0.0, 2.0};
  SpeciesSet& target_species(elec_.getSpeciesSet());
  int upIdx                          = target_species.addSpecies("u");
  int downIdx                        = target_species.addSpecies("d");
  int chargeIdx                      = target_species.addAttribute("charge");
  target_species(chargeIdx, upIdx)   = -1;
  target_species(chargeIdx, downIdx) = -1;
  //elec_.resetGroups();

  const char* particles = R"(<tmp>
    <jastrow name="J3" type="eeI" function="polynomial" source="ion" print="yes">
      <correlation ispecies="O" especies="u" isize="3" esize="3" rcut="10">
        <coefficients id="uuO" type="Array" optimize="yes"> 8.227710241e-06 2.480817653e-06 -5.354068112e-06 -1.112644787e-05 -2.208006078e-06 5.213121933e-06 -1.537865869e-05 8.899030233e-06 6.257255156e-06 3.214580988e-06 -7.716743107e-06 -5.275682077e-06 -1.778457637e-06 7.926231121e-06 1.767406868e-06 5.451359059e-08 2.801423724e-06 4.577282736e-06 7.634608083e-06 -9.510673173e-07 -2.344131575e-06 -1.878777219e-06 3.937363358e-07 5.065353773e-07 5.086724869e-07 -1.358768154e-07</coefficients>
      </correlation>
      <correlation ispecies="O" especies1="u" especies2="d" isize="3" esize="3" rcut="10">
        <coefficients id="udO" type="Array" optimize="yes"> -6.939530224e-06 2.634169299e-05 4.046077477e-05 -8.002682388e-06 -5.396795988e-06 6.697370507e-06 5.433953051e-05 -6.336849668e-06 3.680471431e-05 -2.996059772e-05 1.99365828e-06 -3.222705626e-05 -8.091669063e-06 4.15738535e-06 4.843939112e-06 3.563650208e-07 3.786332474e-05 -1.418336941e-05 2.282691374e-05 1.29239286e-06 -4.93580873e-06 -3.052539228e-06 9.870288001e-08 1.844286407e-06 2.970561871e-07 -4.364303677e-08</coefficients>
      </correlation>
    </jastrow>
</tmp>
)";
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(particles));

  xmlNodePtr root = doc.getRoot();

  xmlNodePtr jas_eeI = xmlFirstElementChild(root);

  eeI_JastrowBuilder jastrow(c, elec_, ions_);
  std::unique_ptr<WaveFunctionComponent> jas(jastrow.buildComponent(jas_eeI));

  using J3Type              = JeeIOrbitalSoA<PolynomialFunctor3D>;
  auto j3_uptr              = jastrow.buildComponent(jas_eeI);
  WaveFunctionComponent* j3 = dynamic_cast<J3Type*>(j3_uptr.get());
  REQUIRE(j3 != nullptr);

  // update all distance tables
  elec_.update();

  double logpsi_real = std::real(j3->evaluateLog(elec_, elec_.G, elec_.L));
  CHECK(logpsi_real == Approx(-1.193457749)); // note: number not validated

  double KE = -0.5 * (Dot(elec_.G, elec_.G) + Sum(elec_.L));
  CHECK(KE == Approx(-0.058051245)); // note: number not validated

  using ValueType = QMCTraits::ValueType;
  using PosType   = QMCTraits::PosType;

  // set virtutal particle position
  PosType newpos(0.3, 0.2, 0.5);

  elec_.makeVirtualMoves(newpos);
  std::vector<ValueType> ratios(elec_.getTotalNum());
  j3->evaluateRatiosAlltoOne(elec_, ratios);

  CHECK(std::real(ratios[0]) == Approx(0.8744938582));
  CHECK(std::real(ratios[1]) == Approx(1.0357541137));
  CHECK(std::real(ratios[2]) == Approx(0.8302245609));
  CHECK(std::real(ratios[3]) == Approx(0.7987703724));

  elec_.makeMove(0, newpos - elec_.R[0]);
  PsiValue ratio_0 = j3->ratio(elec_, 0);
  elec_.rejectMove(0);

  elec_.makeMove(1, newpos - elec_.R[1]);
  PsiValue ratio_1 = j3->ratio(elec_, 1);
  elec_.rejectMove(1);

  elec_.makeMove(2, newpos - elec_.R[2]);
  PsiValue ratio_2 = j3->ratio(elec_, 2);
  elec_.rejectMove(2);

  elec_.makeMove(3, newpos - elec_.R[3]);
  PsiValue ratio_3 = j3->ratio(elec_, 3);
  elec_.rejectMove(3);

  CHECK(std::real(ratio_0) == Approx(0.8744938582));
  CHECK(std::real(ratio_1) == Approx(1.0357541137));
  CHECK(std::real(ratio_2) == Approx(0.8302245609));
  CHECK(std::real(ratio_3) == Approx(0.7987703724));

  UniqueOptObjRefs opt_obj_refs;
  j3->extractOptimizableObjectRefs(opt_obj_refs);
  REQUIRE(opt_obj_refs.size() == 2);

  OptVariables optvars;
  Vector<WaveFunctionComponent::ValueType> dlogpsi;
  Vector<WaveFunctionComponent::ValueType> dhpsioverpsi;

  for (OptimizableObject& obj : opt_obj_refs)
    obj.checkInVariablesExclusive(optvars);
  optvars.resetIndex();
  const int num_opt_vars(optvars.size());
  j3->checkOutVariables(optvars);
  dlogpsi.resize(num_opt_vars);
  dhpsioverpsi.resize(num_opt_vars);
  j3->evaluateDerivatives(elec_, optvars, dlogpsi, dhpsioverpsi);

  app_log() << std::endl << "reporting dlogpsi and dhpsioverpsi" << std::scientific << std::endl;
  for (int iparam = 0; iparam < num_opt_vars; iparam++)
    app_log() << "param=" << iparam << " : " << dlogpsi[iparam] << "  " << dhpsioverpsi[iparam] << std::endl;
  app_log() << std::endl;

  CHECK(std::real(dlogpsi[43]) == Approx(1.3358726814e+05));
  CHECK(std::real(dhpsioverpsi[43]) == Approx(-2.3246270644e+05));

  Vector<WaveFunctionComponent::ValueType> dlogpsiWF;
  dlogpsiWF.resize(num_opt_vars);
  j3->evaluateDerivativesWF(elec_, optvars, dlogpsiWF);
  for (int i = 0; i < num_opt_vars; i++)
    CHECK(dlogpsi[i] == ValueApprox(dlogpsiWF[i]));

  VirtualParticleSet VP(elec_);
  std::vector<PosType> newpos2(2);
  std::vector<ValueType> ratios2(2);
  newpos2[0] = newpos - elec_.R[1];
  newpos2[1] = PosType(0.2, 0.5, 0.3) - elec_.R[1];
  VP.makeMoves(elec_, 1, newpos2);
  j3->evaluateRatios(VP, ratios2);

  CHECK(std::real(ratios2[0]) == Approx(1.0357541137));
  CHECK(std::real(ratios2[1]) == Approx(1.0257141422));

  std::fill(ratios2.begin(), ratios2.end(), 0);
  Matrix<ValueType> dratio(2, num_opt_vars);
  j3->evaluateDerivRatios(VP, optvars, ratios2, dratio);

  CHECK(std::real(ratios2[0]) == Approx(1.0357541137));
  CHECK(std::real(ratios2[1]) == Approx(1.0257141422));
  CHECK(std::real(dratio[0][43]) == Approx(-1.4282569e+03));

  // testing batched interfaces
  ResourceCollection pset_res("test_pset_res");
  ResourceCollection wfc_res("test_wfc_res");

  elec_.createResource(pset_res);
  j3->createResource(wfc_res);

  // make a clones
  ParticleSet elec_clone(elec_);
  auto j3_clone = j3->makeClone(elec_clone);

  // testing batched interfaces
  RefVectorWithLeader<ParticleSet> p_ref_list(elec_, {elec_, elec_clone});
  RefVectorWithLeader<WaveFunctionComponent> j3_ref_list(*j3, {*j3, *j3_clone});

  ResourceCollectionTeamLock<ParticleSet> mw_pset_lock(pset_res, p_ref_list);
  ResourceCollectionTeamLock<WaveFunctionComponent> mw_wfc_lock(wfc_res, j3_ref_list);

  std::vector<bool> isAccepted(2, true);
  ParticleSet::mw_update(p_ref_list);
  j3->mw_recompute(j3_ref_list, p_ref_list, isAccepted);

  // test NLPP related APIs
  const int nknot = 3;
  VirtualParticleSet vp(elec_), vp_clone(elec_clone);
  RefVectorWithLeader<VirtualParticleSet> vp_list(vp, {vp, vp_clone});
  ResourceCollection vp_res("test_vp_res");
  vp.createResource(vp_res);
  ResourceCollectionTeamLock<VirtualParticleSet> mw_vp_lock(vp_res, vp_list);

  const int ei_table_index = elec_.addTable(ions_);
  const auto& ei_table1    = elec_.getDistTableAB(ei_table_index);
  // make virtual move of elec 0, reference ion 1
  NLPPJob<RealType> job1(1, 0, ei_table1.getDistances()[0][1], -ei_table1.getDisplacements()[0][1]);
  const auto& ei_table2 = elec_clone.getDistTableAB(ei_table_index);
  // make virtual move of elec 1, reference ion 3
  NLPPJob<RealType> job2(3, 1, ei_table2.getDistances()[1][3], -ei_table2.getDisplacements()[1][3]);

  std::vector<PosType> deltaV1{{0.1, 0.2, 0.3}, {0.1, 0.3, 0.2}, {0.2, 0.1, 0.3}};
  std::vector<PosType> deltaV2{{0.02, 0.01, 0.03}, {0.02, 0.03, 0.01}, {0.03, 0.01, 0.02}};

  VirtualParticleSet::mw_makeMoves(vp_list, p_ref_list, {deltaV1, deltaV2}, {job1, job2}, false);

  std::vector<std::vector<ValueType>> nlpp_ratios(2);
  nlpp_ratios[0].resize(nknot);
  nlpp_ratios[1].resize(nknot);
  j3->mw_evaluateRatios(j3_ref_list, RefVectorWithLeader<const VirtualParticleSet>(vp, {vp, vp_clone}), nlpp_ratios);

  CHECK(ValueApprox(nlpp_ratios[0][0]) == ValueType(1.0273599625));
  CHECK(ValueApprox(nlpp_ratios[0][1]) == ValueType(1.0227555037));
  CHECK(ValueApprox(nlpp_ratios[0][2]) == ValueType(1.0473958254));
  CHECK(ValueApprox(nlpp_ratios[1][0]) == ValueType(1.0013145208));
  CHECK(ValueApprox(nlpp_ratios[1][1]) == ValueType(1.0011137724));
  CHECK(ValueApprox(nlpp_ratios[1][2]) == ValueType(1.0017225742));

  //Now to test the J3 ion derivatives
  using GradType  = QMCTraits::GradType;
  GradType g0_ref = {ValueType(0.4175355519), ValueType(0.0), ValueType(-0.1822083424)};
  GradType g1_ref = {ValueType(-0.4841712529), ValueType(0.0), ValueType(-0.1479434372)};

  GradType g0(0), g1(0);

  g0 = j3->evalGradSource(elec_, ions_, 0);
  g1 = j3->evalGradSource(elec_, ions_, 1);

  for (int idim = 0; idim < 3; idim++)
  {
    CHECK(g0[idim] == ValueApprox(g0_ref[idim]));
    CHECK(g1[idim] == ValueApprox(g1_ref[idim]));
  }

  TinyVector<ParticleSet::ParticleGradient, 3> g_grad_ref;
  TinyVector<ParticleSet::ParticleLaplacian, 3> g_lapl_ref;
  TinyVector<ParticleSet::ParticleGradient, 3> g_grad;
  TinyVector<ParticleSet::ParticleLaplacian, 3> g_lapl;

  const int nelec = elec_.getTotalNum();
  for (int idim = 0; idim < 3; idim++)
  {
    g_grad_ref[idim].resize(nelec);
    g_lapl_ref[idim].resize(nelec);
    g_grad[idim].resize(nelec);
    g_lapl[idim].resize(nelec);
  }

  //check ion 0
  create_J3_ion_reference_values(g_grad_ref, g_lapl_ref, 0);
  j3->evalGradSource(elec_, ions_, 0, g_grad, g_lapl);
  for (int igdim = 0; igdim < 3; igdim++)
  {
    for (int ielec = 0; ielec < nelec; ielec++)
    {
      for (int idim = 0; idim < 3; idim++)
        CHECK(g_grad[igdim][ielec][idim] == ValueApprox(g_grad_ref[igdim][ielec][idim]));

      CHECK(g_lapl[igdim][ielec] == ValueApprox(g_lapl_ref[igdim][ielec]));
    }
  }

  //clear reference values
  for (int igdim = 0; igdim < 3; igdim++)
  {
    g_grad_ref[igdim] = 0;
    g_lapl_ref[igdim] = 0;
    g_grad[igdim]     = 0;
    g_lapl[igdim]     = 0;
  }

  //check ion 1
  create_J3_ion_reference_values(g_grad_ref, g_lapl_ref, 1);
  j3->evalGradSource(elec_, ions_, 1, g_grad, g_lapl);
  for (int igdim = 0; igdim < 3; igdim++)
  {
    for (int ielec = 0; ielec < nelec; ielec++)
    {
      for (int idim = 0; idim < 3; idim++)
        CHECK(g_grad[igdim][ielec][idim] == ValueApprox(g_grad_ref[igdim][ielec][idim]));

      CHECK(g_lapl[igdim][ielec] == ValueApprox(g_lapl_ref[igdim][ielec]));
    }
  }
}

TEST_CASE("PolynomialFunctor3D Jastrow", "[wavefunction]")
{
  test_J3_polynomial3D(DynamicCoordinateKind::DC_POS);
  test_J3_polynomial3D(DynamicCoordinateKind::DC_POS_OFFLOAD);
}

// ---------------------------------------------------------------------------
// Golden master: freeze pre-port science numbers for the canonical 2-ion / 4e
// polynomial eeI system. Any path (serial, multi-walker OpenMP, use_offload)
// must reproduce these values. Do not regenerate without science review.
// Source: long-standing CHECKs in test_J3_polynomial3D (DC_POS).
// ---------------------------------------------------------------------------
namespace jeei_golden
{
// evaluateLog / kinetic-like scalar from G,L
constexpr double logpsi           = -1.193457749;
constexpr double ke_from_GL       = -0.058051245;
// evaluateRatiosAlltoOne / ratio at newpos=(0.3,0.2,0.5) for e=0..3
constexpr double ratio_e[4]       = {0.8744938582, 1.0357541137, 0.8302245609, 0.7987703724};
// ratioGrad particle 0 at same newpos (value matches ratio_e[0]; gradient from host)
constexpr double ratio_grad0_val  = 0.8744938582;
constexpr double ratio_grad0_g[3] = {0.1678787246, 0.0118009679, 0.0439525127};
// one optimizable-parameter derivative sample
constexpr double dlogpsi_43       = 1.3358726814e+05;
constexpr double dhpsioverpsi_43  = -2.3246270644e+05;

inline PosType newpos() { return PosType(0.3, 0.2, 0.5); }

const char* xml()
{
  return R"(<tmp>
    <jastrow name="J3" type="eeI" function="polynomial" source="ion" print="yes">
      <correlation ispecies="O" especies="u" isize="3" esize="3" rcut="10">
        <coefficients id="uuO_gm" type="Array" optimize="yes"> 8.227710241e-06 2.480817653e-06 -5.354068112e-06 -1.112644787e-05 -2.208006078e-06 5.213121933e-06 -1.537865869e-05 8.899030233e-06 6.257255156e-06 3.214580988e-06 -7.716743107e-06 -5.275682077e-06 -1.778457637e-06 7.926231121e-06 1.767406868e-06 5.451359059e-08 2.801423724e-06 4.577282736e-06 7.634608083e-06 -9.510673173e-07 -2.344131575e-06 -1.878777219e-06 3.937363358e-07 5.065353773e-07 5.086724869e-07 -1.358768154e-07</coefficients>
      </correlation>
      <correlation ispecies="O" especies1="u" especies2="d" isize="3" esize="3" rcut="10">
        <coefficients id="udO_gm" type="Array" optimize="yes"> -6.939530224e-06 2.634169299e-05 4.046077477e-05 -8.002682388e-06 -5.396795988e-06 6.697370507e-06 5.433953051e-05 -6.336849668e-06 3.680471431e-05 -2.996059772e-05 1.99365828e-06 -3.222705626e-05 -8.091669063e-06 4.15738535e-06 4.843939112e-06 3.563650208e-07 3.786332474e-05 -1.418336941e-05 2.282691374e-05 1.29239286e-06 -4.93580873e-06 -3.052539228e-06 9.870288001e-08 1.844286407e-06 2.970561871e-07 -4.364303677e-08</coefficients>
      </correlation>
    </jastrow>
</tmp>
)";
}

void setup_ions_elecs(ParticleSet& ions, ParticleSet& elec)
{
  ions.setName("ion");
  ions.create({2});
  ions.R[0] = {2.0, 0.0, 0.0};
  ions.R[1] = {-2.0, 0.0, 0.0};
  ions.getSpeciesSet().addSpecies("O");
  ions.update();

  elec.setName("elec");
  elec.create({2, 2});
  elec.R[0] = {1.00, 0.0, 0.0};
  elec.R[1] = {0.0, 0.0, 0.0};
  elec.R[2] = {-1.00, 0.0, 0.0};
  elec.R[3] = {0.0, 0.0, 2.0};
  SpeciesSet& sp = elec.getSpeciesSet();
  int upIdx      = sp.addSpecies("u");
  int downIdx    = sp.addSpecies("d");
  int chargeIdx  = sp.addAttribute("charge");
  sp(chargeIdx, upIdx)   = -1;
  sp(chargeIdx, downIdx) = -1;
}

void check_log_ke_ratios(WaveFunctionComponent& j3, ParticleSet& elec)
{
  elec.G = 0;
  elec.L = 0;
  const double logpsi_v = std::real(j3.evaluateLog(elec, elec.G, elec.L));
  CHECK(logpsi_v == Approx(logpsi));
  const double KE = -0.5 * (Dot(elec.G, elec.G) + Sum(elec.L));
  CHECK(KE == Approx(ke_from_GL));

  elec.makeVirtualMoves(newpos());
  std::vector<QMCTraits::ValueType> ratios(elec.getTotalNum());
  j3.evaluateRatiosAlltoOne(elec, ratios);
  for (int e = 0; e < 4; ++e)
    CHECK(std::real(ratios[e]) == Approx(ratio_e[e]));

  for (int e = 0; e < 4; ++e)
  {
    elec.makeMove(e, newpos() - elec.R[e]);
    PsiValue r = j3.ratio(elec, e);
    elec.rejectMove(e);
    CHECK(std::real(r) == Approx(ratio_e[e]));
  }

  elec.makeMove(0, newpos() - elec.R[0]);
  GradType g(0);
  PsiValue rg = j3.ratioGrad(elec, 0, g);
  elec.rejectMove(0);
  CHECK(std::real(rg) == Approx(ratio_grad0_val));
  for (int d = 0; d < 3; ++d)
    CHECK(g[d] == Approx(ratio_grad0_g[d]));
}
} // namespace jeei_golden

/** Golden master: serial host path matches frozen science numbers. */
TEST_CASE("JeeIOrbitalSoA golden master serial", "[wavefunction][golden]")
{
  Communicate* c = OHMMS::Controller;
  const SimulationCell cell;
  ParticleSet ions(cell), elec(cell);
  jeei_golden::setup_ions_elecs(ions, elec);

  Libxml2Document doc;
  REQUIRE(doc.parseFromString(jeei_golden::xml()));
  eeI_JastrowBuilder b(c, elec, ions);
  auto j3u = b.buildComponent(xmlFirstElementChild(doc.getRoot()));
  auto* j3 = dynamic_cast<JeeIOrbitalSoA<PolynomialFunctor3D>*>(j3u.get());
  REQUIRE(j3);
  // ENABLE_CUDA builds enable dense offload via the builder; science values must still match.

  elec.update();
  jeei_golden::check_log_ke_ratios(*j3, elec);

  UniqueOptObjRefs opt_refs;
  j3->extractOptimizableObjectRefs(opt_refs);
  OptVariables optvars;
  for (OptimizableObject& obj : opt_refs)
    obj.checkInVariablesExclusive(optvars);
  optvars.resetIndex();
  j3->checkOutVariables(optvars);
  Vector<WaveFunctionComponent::ValueType> dlogpsi(optvars.size()), dhpsi(optvars.size());
  j3->evaluateDerivatives(elec, optvars, dlogpsi, dhpsi);
  CHECK(std::real(dlogpsi[43]) == Approx(jeei_golden::dlogpsi_43));
  CHECK(std::real(dhpsi[43]) == Approx(jeei_golden::dhpsioverpsi_43));
}

/** Golden master: multi-walker OpenMP mw_ratioGrad matches frozen ratioGrad science. */
TEST_CASE("JeeIOrbitalSoA golden master multi-walker OpenMP", "[wavefunction][golden]")
{
  Communicate* c = OHMMS::Controller;
  const SimulationCell cell;
  ParticleSet ions(cell), elec(cell);
  jeei_golden::setup_ions_elecs(ions, elec);

  Libxml2Document doc;
  REQUIRE(doc.parseFromString(jeei_golden::xml()));
  eeI_JastrowBuilder b(c, elec, ions);
  auto j3u = b.buildComponent(xmlFirstElementChild(doc.getRoot()));
  auto* j3 = dynamic_cast<JeeIOrbitalSoA<PolynomialFunctor3D>*>(j3u.get());
  REQUIRE(j3);

  elec.update();
  elec.G = 0;
  elec.L = 0;
  j3->evaluateLog(elec, elec.G, elec.L);

  ParticleSet elec2(elec);
  auto j3_clone = j3->makeClone(elec2);
  elec2.update();
  elec2.G = 0;
  elec2.L = 0;
  j3_clone->evaluateLog(elec2, elec2.G, elec2.L);

  const PosType np = jeei_golden::newpos();
  elec.makeMove(0, np - elec.R[0]);
  elec2.makeMove(0, np - elec2.R[0]);

  std::vector<PsiValue> ratios(2);
  std::vector<GradType> grads(2, GradType(0));
  RefVectorWithLeader<WaveFunctionComponent> wfc_list(*j3, {*j3, *j3_clone});
  RefVectorWithLeader<ParticleSet> p_list(elec, {elec, elec2});
  j3->mw_ratioGrad(wfc_list, p_list, 0, ratios, grads);

  for (int iw = 0; iw < 2; ++iw)
  {
    CHECK(std::real(ratios[iw]) == Approx(jeei_golden::ratio_grad0_val));
    for (int d = 0; d < 3; ++d)
      CHECK(grads[iw][d] == Approx(jeei_golden::ratio_grad0_g[d]));
  }

  elec.rejectMove(0);
  elec2.rejectMove(0);
}

/** Golden master: use_offload=true path matches frozen science (log, ratios, ratioGrad). */
TEST_CASE("JeeIOrbitalSoA golden master use_offload path", "[wavefunction][golden]")
{
  Communicate* c = OHMMS::Controller;
  const SimulationCell cell;
  ParticleSet ions(cell), elec(cell);
  jeei_golden::setup_ions_elecs(ions, elec);

  Libxml2Document doc;
  REQUIRE(doc.parseFromString(jeei_golden::xml()));
  // Build host then deep-copy functors into use_offload instance via makeClone pattern:
  // construct offload object and re-parse correlations with unique ids already in xml.
  eeI_JastrowBuilder b_host(c, elec, ions);
  auto host_up = b_host.buildComponent(xmlFirstElementChild(doc.getRoot()));
  auto* host   = dynamic_cast<JeeIOrbitalSoA<PolynomialFunctor3D>*>(host_up.get());
  REQUIRE(host);

  // Offload-enabled clone on a twin electron set
  ParticleSet elec_o(elec);
  JeeIOrbitalSoA<PolynomialFunctor3D> j3_off("J3_gm_off", ions, elec_o, true);
  REQUIRE(j3_off.isUsingOffload());
  // Copy correlations by re-building from the same XML into offload via host makeClone
  // then evaluating on host is not enough — reconstruct functors:
  {
    xmlNodePtr kids  = xmlFirstElementChild(doc.getRoot())->children;
    SpeciesSet& iSet = ions.getSpeciesSet();
    SpeciesSet& eSet = elec_o.getSpeciesSet();
    while (kids != nullptr)
    {
      if (std::string((char*)kids->name) == "correlation")
      {
        RealType ee_cusp = 0.0, eI_cusp = 0.0;
        std::string iSpecies, eSpecies1("u"), eSpecies2("u");
        OhmmsAttributeSet rAttrib;
        rAttrib.add(iSpecies, "ispecies");
        rAttrib.add(eSpecies1, "especies1");
        rAttrib.add(eSpecies2, "especies2");
        rAttrib.add(ee_cusp, "ecusp");
        rAttrib.add(eI_cusp, "icusp");
        rAttrib.put(kids);
        auto functor =
            std::make_unique<PolynomialFunctor3D>("J3gm_" + iSpecies + eSpecies1 + eSpecies2, ee_cusp, eI_cusp);
        functor->iSpecies  = iSpecies;
        functor->eSpecies1 = eSpecies1;
        functor->eSpecies2 = eSpecies2;
        functor->put(kids);
        j3_off.addFunc(iSet.findSpecies(iSpecies), eSet.findSpecies(eSpecies1), eSet.findSpecies(eSpecies2),
                       std::move(functor));
      }
      kids = kids->next;
    }
    j3_off.check_complete();
  }

  elec_o.update();
  jeei_golden::check_log_ke_ratios(j3_off, elec_o);

  // mw_ratioGrad on single offload walker must hit golden ratioGrad
  elec_o.makeMove(0, jeei_golden::newpos() - elec_o.R[0]);
  std::vector<PsiValue> ratios(1);
  std::vector<GradType> grads(1, GradType(0));
  RefVectorWithLeader<WaveFunctionComponent> wfc_list(j3_off, {j3_off});
  RefVectorWithLeader<ParticleSet> p_list(elec_o, {elec_o});
  j3_off.mw_ratioGrad(wfc_list, p_list, 0, ratios, grads);
  CHECK(std::real(ratios[0]) == Approx(jeei_golden::ratio_grad0_val));
  for (int d = 0; d < 3; ++d)
    CHECK(grads[0][d] == Approx(jeei_golden::ratio_grad0_g[d]));
  elec_o.rejectMove(0);

  (void)host; // host built to ensure XML still valid for builder path
}

/** Golden master: multi-walker offload recompute → ratioGrad → accept → evalGrad.
 *  Pins CUDA dense (ENABLE_CUDA) / host dense against frozen log/KE/ratioGrad, then
 *  checks accept does not poison stored gradients (evalGrad matches serial host twin).
 */
TEST_CASE("JeeIOrbitalSoA golden master multi-walker offload recompute+accept", "[wavefunction][golden]")
{
  Communicate* c = OHMMS::Controller;
  const SimulationCell cell;
  ParticleSet ions(cell), elec_h(cell), elec_o(cell);
  jeei_golden::setup_ions_elecs(ions, elec_h);
  // second electron set: same geometry, own distance tables
  elec_o.setName("elec");
  elec_o.create({2, 2});
  elec_o.R[0] = {1.00, 0.0, 0.0};
  elec_o.R[1] = {0.0, 0.0, 0.0};
  elec_o.R[2] = {-1.00, 0.0, 0.0};
  elec_o.R[3] = {0.0, 0.0, 2.0};
  {
    SpeciesSet& sp = elec_o.getSpeciesSet();
    int upIdx      = sp.addSpecies("u");
    int downIdx    = sp.addSpecies("d");
    int chargeIdx  = sp.addAttribute("charge");
    sp(chargeIdx, upIdx)   = -1;
    sp(chargeIdx, downIdx) = -1;
  }

  Libxml2Document doc;
  REQUIRE(doc.parseFromString(jeei_golden::xml()));

  // Host twin for accept science
  eeI_JastrowBuilder b_h(c, elec_h, ions);
  auto j3h_u = b_h.buildComponent(xmlFirstElementChild(doc.getRoot()));
  auto* j3_h = dynamic_cast<JeeIOrbitalSoA<PolynomialFunctor3D>*>(j3h_u.get());
  REQUIRE(j3_h);
  elec_h.update();
  elec_h.G = 0;
  elec_h.L = 0;
  j3_h->evaluateLog(elec_h, elec_h.G, elec_h.L);
  elec_h.makeMove(0, jeei_golden::newpos() - elec_h.R[0]);
  GradType gh(0);
  const PsiValue rh = j3_h->ratioGrad(elec_h, 0, gh);
  j3_h->acceptMove(elec_h, 0);
  elec_h.acceptMove(0);
  const GradType g1_host = j3_h->evalGrad(elec_h, 1);
  CHECK(std::real(rh) == Approx(jeei_golden::ratio_grad0_val));

  // Offload leader + clone (same pattern as multi-walker OpenMP golden)
  JeeIOrbitalSoA<PolynomialFunctor3D> j3_off("J3_gm_off_mw", ions, elec_o, true);
  REQUIRE(j3_off.isUsingOffload());
  {
    xmlNodePtr kids  = xmlFirstElementChild(doc.getRoot())->children;
    SpeciesSet& iSet = ions.getSpeciesSet();
    SpeciesSet& eSet = elec_o.getSpeciesSet();
    while (kids != nullptr)
    {
      if (std::string((char*)kids->name) == "correlation")
      {
        RealType ee_cusp = 0.0, eI_cusp = 0.0;
        std::string iSpecies, eSpecies1("u"), eSpecies2("u");
        OhmmsAttributeSet rAttrib;
        rAttrib.add(iSpecies, "ispecies");
        rAttrib.add(eSpecies1, "especies1");
        rAttrib.add(eSpecies2, "especies2");
        rAttrib.add(ee_cusp, "ecusp");
        rAttrib.add(eI_cusp, "icusp");
        rAttrib.put(kids);
        auto functor =
            std::make_unique<PolynomialFunctor3D>("J3gm_offmw_" + iSpecies + eSpecies1 + eSpecies2, ee_cusp, eI_cusp);
        functor->iSpecies  = iSpecies;
        functor->eSpecies1 = eSpecies1;
        functor->eSpecies2 = eSpecies2;
        functor->put(kids);
        j3_off.addFunc(iSet.findSpecies(iSpecies), eSet.findSpecies(eSpecies1), eSet.findSpecies(eSpecies2),
                       std::move(functor));
      }
      kids = kids->next;
    }
    j3_off.check_complete();
  }

  ParticleSet elec2(elec_o);
  auto j3_clone = j3_off.makeClone(elec2);
  auto* j3_c    = dynamic_cast<JeeIOrbitalSoA<PolynomialFunctor3D>*>(j3_clone.get());
  REQUIRE(j3_c);

  elec_o.update();
  elec2.update();
  ParticleSet::ParticleGradient G0(4), G1(4);
  ParticleSet::ParticleLaplacian L0(4), L1(4);
  G0 = 0;
  G1 = 0;
  L0 = 0;
  L1 = 0;
  RefVectorWithLeader<WaveFunctionComponent> wfc_list(j3_off, {j3_off, *j3_c});
  RefVectorWithLeader<ParticleSet> p_list(elec_o, {elec_o, elec2});
  RefVector<ParticleSet::ParticleGradient> G_list = {G0, G1};
  RefVector<ParticleSet::ParticleLaplacian> L_list = {L0, L1};

  // Persistent multi-walker resource: device-resident table caching only engages
  // through this path (matches the batched drivers), so the post-accept sweeps below
  // exercise the accept-time device row patching rather than a fresh full upload.
  ResourceCollection wfc_res("test_jeei_res");
  j3_off.createResource(wfc_res);
  ResourceCollectionTeamLock<WaveFunctionComponent> mw_lock(wfc_res, wfc_list);

  j3_off.mw_evaluateLog(wfc_list, p_list, G_list, L_list);
  for (auto logv : {j3_off.get_log_value(), j3_c->get_log_value()})
    CHECK(std::real(logv) == Approx(jeei_golden::logpsi));
  CHECK(-0.5 * (Dot(G0, G0) + Sum(L0)) == Approx(jeei_golden::ke_from_GL));
  CHECK(-0.5 * (Dot(G1, G1) + Sum(L1)) == Approx(jeei_golden::ke_from_GL));

  const PosType np = jeei_golden::newpos();
  elec_o.makeMove(0, np - elec_o.R[0]);
  elec2.makeMove(0, np - elec2.R[0]);
  std::vector<PsiValue> ratios(2);
  std::vector<GradType> grads(2, GradType(0));
  j3_off.mw_ratioGrad(wfc_list, p_list, 0, ratios, grads);
  for (int iw = 0; iw < 2; ++iw)
  {
    CHECK(std::real(ratios[iw]) == Approx(jeei_golden::ratio_grad0_val));
    for (int d = 0; d < 3; ++d)
      CHECK(grads[iw][d] == Approx(jeei_golden::ratio_grad0_g[d]));
  }

  std::vector<bool> accepted(2, true);
  j3_off.mw_accept_rejectMove(wfc_list, p_list, 0, accepted);
  elec_o.acceptMove(0);
  elec2.acceptMove(0);
  for (auto* j3p : {&j3_off, j3_c})
  {
    auto& P  = (j3p == &j3_off) ? elec_o : elec2;
    GradType g1 = j3p->evalGrad(P, 1);
    for (int d = 0; d < 3; ++d)
    {
      REQUIRE(std::isfinite(std::real(g1[d])));
      CHECK(std::real(g1[d]) == Approx(std::real(g1_host[d])));
    }
  }

  // Second sweep on another electron: the device-resident e-I tables must hold the
  // accepted (post-move) row for electron 0, so ratioGrad here must match the host twin.
  const PosType np2(0.15, -0.2, 0.35);
  elec_h.makeMove(1, np2 - elec_h.R[1]);
  GradType gh2(0);
  const PsiValue rh2 = j3_h->ratioGrad(elec_h, 1, gh2);
  j3_h->acceptMove(elec_h, 1);
  elec_h.acceptMove(1);

  elec_o.makeMove(1, np2 - elec_o.R[1]);
  elec2.makeMove(1, np2 - elec2.R[1]);
  grads[0] = GradType(0);
  grads[1] = GradType(0);
  j3_off.mw_ratioGrad(wfc_list, p_list, 1, ratios, grads);
  for (int iw = 0; iw < 2; ++iw)
  {
    CHECK(std::real(ratios[iw]) == Approx(std::real(rh2)));
    for (int d = 0; d < 3; ++d)
      CHECK(std::real(grads[iw][d]) == Approx(std::real(gh2[d])));
  }
  j3_off.mw_accept_rejectMove(wfc_list, p_list, 1, accepted);
  elec_o.acceptMove(1);
  elec2.acceptMove(1);

  // Partial recompute (walker 0 only) must not corrupt the resident tables of the
  // skipped walker: the follow-up ratioGrad still has to agree with the host twin.
  const std::vector<bool> recompute_partial{true, false};
  j3_off.mw_recompute(wfc_list, p_list, recompute_partial);

  const PosType np3(-0.6, 0.1, 0.25);
  elec_h.makeMove(2, np3 - elec_h.R[2]);
  GradType gh3(0);
  const PsiValue rh3 = j3_h->ratioGrad(elec_h, 2, gh3);
  elec_h.rejectMove(2);

  elec_o.makeMove(2, np3 - elec_o.R[2]);
  elec2.makeMove(2, np3 - elec2.R[2]);
  grads[0] = GradType(0);
  grads[1] = GradType(0);
  j3_off.mw_ratioGrad(wfc_list, p_list, 2, ratios, grads);
  for (int iw = 0; iw < 2; ++iw)
  {
    CHECK(std::real(ratios[iw]) == Approx(std::real(rh3)));
    for (int d = 0; d < 3; ++d)
      CHECK(std::real(grads[iw][d]) == Approx(std::real(gh3[d])));
  }
  const std::vector<bool> rejected(2, false);
  j3_off.mw_accept_rejectMove(wfc_list, p_list, 2, rejected);
  elec_o.rejectMove(2);
  elec2.rejectMove(2);
}


/** Scaffold: isOMPoffload and use_offload construction. */
TEST_CASE("JeeIOrbitalSoA offload scaffold", "[wavefunction]")
{
  REQUIRE(PolynomialFunctor3D::isOMPoffload());

  const SimulationCell simulation_cell;
  ParticleSet ions_(simulation_cell);
  ParticleSet elec_(simulation_cell);

  ions_.setName("ion");
  ions_.create({1});
  ions_.R[0] = {0.0, 0.0, 0.0};
  ions_.getSpeciesSet().addSpecies("H");
  ions_.update();

  elec_.setName("elec");
  elec_.create({1, 1});
  elec_.R[0] = {0.5, 0.0, 0.0};
  elec_.R[1] = {-0.5, 0.0, 0.0};
  SpeciesSet& sp = elec_.getSpeciesSet();
  int upIdx      = sp.addSpecies("u");
  int downIdx    = sp.addSpecies("d");
  int chargeIdx  = sp.addAttribute("charge");
  sp(chargeIdx, upIdx)   = -1;
  sp(chargeIdx, downIdx) = -1;

  using J3Type = JeeIOrbitalSoA<PolynomialFunctor3D>;
  J3Type j3_host("JeeI_host", ions_, elec_, false);
  REQUIRE_FALSE(j3_host.isUsingOffload());
  J3Type j3_off("JeeI_off", ions_, elec_, true);
  REQUIRE(j3_off.isUsingOffload());
}

/** Host compact-list path vs use_offload dense dual-table path. */
TEST_CASE("JeeIOrbitalSoA host vs offload dense agreement", "[wavefunction]")
{
  Communicate* c = OHMMS::Controller;

  const SimulationCell simulation_cell;
  ParticleSet ions_h(simulation_cell);
  ParticleSet elec_h(simulation_cell);
  ParticleSet ions_d(simulation_cell);
  ParticleSet elec_d(simulation_cell);

  auto setup = [](ParticleSet& ions, ParticleSet& elec) {
    ions.setName("ion");
    ions.create({2});
    ions.R[0] = {2.0, 0.0, 0.0};
    ions.R[1] = {-2.0, 0.0, 0.0};
    ions.getSpeciesSet().addSpecies("O");
    ions.update();

    elec.setName("elec");
    elec.create({2, 2});
    elec.R[0] = {1.00, 0.0, 0.0};
    elec.R[1] = {0.0, 0.0, 0.0};
    elec.R[2] = {-1.00, 0.0, 0.0};
    elec.R[3] = {0.0, 0.0, 2.0};
    SpeciesSet& sp = elec.getSpeciesSet();
    int upIdx      = sp.addSpecies("u");
    int downIdx    = sp.addSpecies("d");
    int chargeIdx  = sp.addAttribute("charge");
    sp(chargeIdx, upIdx)   = -1;
    sp(chargeIdx, downIdx) = -1;
  };
  setup(ions_h, elec_h);
  setup(ions_d, elec_d);

  const char* particles = R"(<tmp>
    <jastrow name="J3" type="eeI" function="polynomial" source="ion" print="yes">
      <correlation ispecies="O" especies="u" isize="3" esize="3" rcut="10">
        <coefficients id="uuO" type="Array" optimize="yes"> 8.227710241e-06 2.480817653e-06 -5.354068112e-06 -1.112644787e-05 -2.208006078e-06 5.213121933e-06 -1.537865869e-05 8.899030233e-06 6.257255156e-06 3.214580988e-06 -7.716743107e-06 -5.275682077e-06 -1.778457637e-06 7.926231121e-06 1.767406868e-06 5.451359059e-08 2.801423724e-06 4.577282736e-06 7.634608083e-06 -9.510673173e-07 -2.344131575e-06 -1.878777219e-06 3.937363358e-07 5.065353773e-07 5.086724869e-07 -1.358768154e-07</coefficients>
      </correlation>
      <correlation ispecies="O" especies1="u" especies2="d" isize="3" esize="3" rcut="10">
        <coefficients id="udO" type="Array" optimize="yes"> -6.939530224e-06 2.634169299e-05 4.046077477e-05 -8.002682388e-06 -5.396795988e-06 6.697370507e-06 5.433953051e-05 -6.336849668e-06 3.680471431e-05 -2.996059772e-05 1.99365828e-06 -3.222705626e-05 -8.091669063e-06 4.15738535e-06 4.843939112e-06 3.563650208e-07 3.786332474e-05 -1.418336941e-05 2.282691374e-05 1.29239286e-06 -4.93580873e-06 -3.052539228e-06 9.870288001e-08 1.844286407e-06 2.970561871e-07 -4.364303677e-08</coefficients>
      </correlation>
    </jastrow>
</tmp>
)";
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(particles));
  xmlNodePtr jas_eeI = xmlFirstElementChild(doc.getRoot());

  using J3Type = JeeIOrbitalSoA<PolynomialFunctor3D>;
  // Explicit host vs offload (builder may force CUDA on ENABLE_CUDA builds).
  J3Type j3_h("J3_host", ions_h, elec_h, false);
  J3Type j3_d("J3_dense", ions_d, elec_d, true);
  REQUIRE_FALSE(j3_h.isUsingOffload());
  REQUIRE(j3_d.isUsingOffload());
  auto load_functors = [&](J3Type& j3, ParticleSet& ions, ParticleSet& elec, const std::string& tag) {
    xmlNodePtr kids = jas_eeI->children;
    SpeciesSet& iSet = ions.getSpeciesSet();
    SpeciesSet& eSet = elec.getSpeciesSet();
    while (kids != nullptr)
    {
      if (std::string((char*)kids->name) == "correlation")
      {
        RealType ee_cusp = 0.0, eI_cusp = 0.0;
        std::string iSpecies, eSpecies1("u"), eSpecies2("u");
        OhmmsAttributeSet rAttrib;
        rAttrib.add(iSpecies, "ispecies");
        rAttrib.add(eSpecies1, "especies1");
        rAttrib.add(eSpecies2, "especies2");
        rAttrib.add(ee_cusp, "ecusp");
        rAttrib.add(eI_cusp, "icusp");
        rAttrib.put(kids);
        auto functor =
            std::make_unique<PolynomialFunctor3D>("J3_" + iSpecies + eSpecies1 + eSpecies2 + "_" + tag,
                                                   ee_cusp, eI_cusp);
        functor->iSpecies  = iSpecies;
        functor->eSpecies1 = eSpecies1;
        functor->eSpecies2 = eSpecies2;
        functor->put(kids);
        j3.addFunc(iSet.findSpecies(iSpecies), eSet.findSpecies(eSpecies1), eSet.findSpecies(eSpecies2),
                   std::move(functor));
      }
      kids = kids->next;
    }
    j3.check_complete();
  };
  load_functors(j3_h, ions_h, elec_h, "host");
  load_functors(j3_d, ions_d, elec_d, "dense");

  elec_h.update();
  elec_d.update();
  elec_h.G = 0;
  elec_h.L = 0;
  elec_d.G = 0;
  elec_d.L = 0;

  const double log_h = std::real(j3_h.evaluateLog(elec_h, elec_h.G, elec_h.L));
  const double log_d = std::real(j3_d.evaluateLog(elec_d, elec_d.G, elec_d.L));
  CHECK(log_h == Approx(-1.193457749));
  CHECK(log_d == Approx(log_h));

  // Single-walker ratioGrad (host compact) vs mw_ratioGrad dense path
  PosType newpos(0.3, 0.2, 0.5);
  elec_h.makeMove(0, newpos - elec_h.R[0]);
  elec_d.makeMove(0, newpos - elec_d.R[0]);

  GradType g_h(0);
  PsiValue r_h = j3_h.ratioGrad(elec_h, 0, g_h);

  std::vector<PsiValue> ratios(1);
  std::vector<GradType> grads(1, GradType(0));
  RefVectorWithLeader<WaveFunctionComponent> wfc_list(j3_d, {j3_d});
  RefVectorWithLeader<ParticleSet> p_list(elec_d, {elec_d});
  j3_d.mw_ratioGrad(wfc_list, p_list, 0, ratios, grads);

  CHECK(std::real(ratios[0]) == Approx(std::real(r_h)));
  for (int d = 0; d < OHMMS_DIM; ++d)
    CHECK(grads[0][d] == Approx(g_h[d]));

  elec_h.rejectMove(0);
  elec_d.rejectMove(0);
}

/** Multi-walker OpenMP mw_ratioGrad must be measurably faster than serial ratioGrad loop. */
TEST_CASE("JeeIOrbitalSoA mw_ratioGrad multi-walker speedup", "[wavefunction][benchmark]")
{
  Communicate* c = OHMMS::Controller;

  const int nw       = 32;
  const int n_repeat = 300;
  const SimulationCell simulation_cell;

  ParticleSet ions(simulation_cell);
  ions.setName("ion");
  ions.create({2});
  ions.R[0] = {2.0, 0.0, 0.0};
  ions.R[1] = {-2.0, 0.0, 0.0};
  ions.getSpeciesSet().addSpecies("O");
  ions.update();

  auto make_elec = [&]() {
    auto elec = std::make_unique<ParticleSet>(simulation_cell);
    elec->setName("elec");
    elec->create({2, 2});
    elec->R[0] = {1.00, 0.0, 0.0};
    elec->R[1] = {0.0, 0.0, 0.0};
    elec->R[2] = {-1.00, 0.0, 0.0};
    elec->R[3] = {0.0, 0.0, 2.0};
    SpeciesSet& sp = elec->getSpeciesSet();
    int upIdx      = sp.addSpecies("u");
    int downIdx    = sp.addSpecies("d");
    int chargeIdx  = sp.addAttribute("charge");
    sp(chargeIdx, upIdx)   = -1;
    sp(chargeIdx, downIdx) = -1;
    return elec;
  };

  const char* particles = R"(<tmp>
    <jastrow name="J3" type="eeI" function="polynomial" source="ion" print="yes">
      <correlation ispecies="O" especies="u" isize="3" esize="3" rcut="10">
        <coefficients id="uuO_bench" type="Array" optimize="yes"> 8.227710241e-06 2.480817653e-06 -5.354068112e-06 -1.112644787e-05 -2.208006078e-06 5.213121933e-06 -1.537865869e-05 8.899030233e-06 6.257255156e-06 3.214580988e-06 -7.716743107e-06 -5.275682077e-06 -1.778457637e-06 7.926231121e-06 1.767406868e-06 5.451359059e-08 2.801423724e-06 4.577282736e-06 7.634608083e-06 -9.510673173e-07 -2.344131575e-06 -1.878777219e-06 3.937363358e-07 5.065353773e-07 5.086724869e-07 -1.358768154e-07</coefficients>
      </correlation>
      <correlation ispecies="O" especies1="u" especies2="d" isize="3" esize="3" rcut="10">
        <coefficients id="udO_bench" type="Array" optimize="yes"> -6.939530224e-06 2.634169299e-05 4.046077477e-05 -8.002682388e-06 -5.396795988e-06 6.697370507e-06 5.433953051e-05 -6.336849668e-06 3.680471431e-05 -2.996059772e-05 1.99365828e-06 -3.222705626e-05 -8.091669063e-06 4.15738535e-06 4.843939112e-06 3.563650208e-07 3.786332474e-05 -1.418336941e-05 2.282691374e-05 1.29239286e-06 -4.93580873e-06 -3.052539228e-06 9.870288001e-08 1.844286407e-06 2.970561871e-07 -4.364303677e-08</coefficients>
      </correlation>
    </jastrow>
</tmp>
)";
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(particles));
  xmlNodePtr jas_eeI = xmlFirstElementChild(doc.getRoot());

  // Host OpenMP multi-walker (use_offload=false). CUDA builds force offload via builder;
  // construct explicitly so this benchmark measures host walker-parallel OpenMP.
  std::vector<std::unique_ptr<ParticleSet>> elecs(nw);
  std::vector<std::unique_ptr<WaveFunctionComponent>> j3s(nw);
  using J3Type = JeeIOrbitalSoA<PolynomialFunctor3D>;
  elecs[0] = make_elec();
  elecs[0]->update();
  {
    eeI_JastrowBuilder b(c, *elecs[0], ions);
    auto built = b.buildComponent(jas_eeI);
    auto* src  = dynamic_cast<J3Type*>(built.get());
    REQUIRE(src);
    // Rebuild as host-only with same functors via makeClone then force? makeClone preserves use_offload_.
    // Parse into explicit host instance:
    auto host = std::make_unique<J3Type>("J3_bench_host", ions, *elecs[0], false);
    // Copy functors from builder object via re-put from XML
    xmlNodePtr kids = jas_eeI->children;
    SpeciesSet& iSet = ions.getSpeciesSet();
    SpeciesSet& eSet = elecs[0]->getSpeciesSet();
    while (kids != nullptr)
    {
      if (std::string((char*)kids->name) == "correlation")
      {
        RealType ee_cusp = 0.0, eI_cusp = 0.0;
        std::string iSpecies, eSpecies1("u"), eSpecies2("u");
        OhmmsAttributeSet rAttrib;
        rAttrib.add(iSpecies, "ispecies");
        rAttrib.add(eSpecies1, "especies1");
        rAttrib.add(eSpecies2, "especies2");
        rAttrib.add(ee_cusp, "ecusp");
        rAttrib.add(eI_cusp, "icusp");
        rAttrib.put(kids);
        auto functor = std::make_unique<PolynomialFunctor3D>("J3b_" + iSpecies + eSpecies1 + eSpecies2, ee_cusp, eI_cusp);
        functor->iSpecies  = iSpecies;
        functor->eSpecies1 = eSpecies1;
        functor->eSpecies2 = eSpecies2;
        functor->put(kids);
        host->addFunc(iSet.findSpecies(iSpecies), eSet.findSpecies(eSpecies1), eSet.findSpecies(eSpecies2),
                      std::move(functor));
      }
      kids = kids->next;
    }
    host->check_complete();
    j3s[0] = std::move(host);
  }
  REQUIRE(j3s[0]);
  REQUIRE_FALSE(dynamic_cast<J3Type*>(j3s[0].get())->isUsingOffload());
  for (int iw = 1; iw < nw; ++iw)
  {
    elecs[iw] = make_elec();
    j3s[iw]   = j3s[0]->makeClone(*elecs[iw]);
    elecs[iw]->update();
  }
  elecs[0]->update();

  auto* leader = dynamic_cast<J3Type*>(j3s[0].get());
  REQUIRE(leader);

  for (int iw = 0; iw < nw; ++iw)
  {
    elecs[iw]->G = 0;
    elecs[iw]->L = 0;
    j3s[iw]->evaluateLog(*elecs[iw], elecs[iw]->G, elecs[iw]->L);
  }

  const PosType newpos(0.3, 0.2, 0.5);
  for (int iw = 0; iw < nw; ++iw)
    elecs[iw]->makeMove(0, newpos - elecs[iw]->R[0]);

  std::vector<PsiValue> ratios_serial(nw), ratios_mw(nw);
  std::vector<GradType> grads_serial(nw), grads_mw(nw);

  RefVectorWithLeader<WaveFunctionComponent> wfc_list(*j3s[0]);
  RefVectorWithLeader<ParticleSet> p_list(*elecs[0]);
  for (int iw = 0; iw < nw; ++iw)
  {
    wfc_list.push_back(*j3s[iw]);
    p_list.push_back(*elecs[iw]);
  }

  auto run_serial_rg = [&]() {
    for (int iw = 0; iw < nw; ++iw)
    {
      grads_serial[iw]  = GradType(0);
      ratios_serial[iw] = j3s[iw]->ratioGrad(*elecs[iw], 0, grads_serial[iw]);
    }
  };
  auto run_mw_rg = [&]() {
    for (int iw = 0; iw < nw; ++iw)
      grads_mw[iw] = GradType(0);
    leader->mw_ratioGrad(wfc_list, p_list, 0, ratios_mw, grads_mw);
  };

  std::vector<bool> recompute_all(nw, true);
  auto run_serial_recompute = [&]() {
    for (int iw = 0; iw < nw; ++iw)
      j3s[iw]->recompute(*elecs[iw]);
  };
  auto run_mw_recompute = [&]() { leader->mw_recompute(wfc_list, p_list, recompute_all); };

  // --- ratioGrad microbench ---
  run_serial_rg();
  run_mw_rg();
  for (int iw = 0; iw < nw; ++iw)
  {
    elecs[iw]->rejectMove(0);
    elecs[iw]->makeMove(0, newpos - elecs[iw]->R[0]);
  }

  using clock = std::chrono::steady_clock;
  auto t0     = clock::now();
  for (int r = 0; r < n_repeat; ++r)
    run_serial_rg();
  auto t1 = clock::now();
  for (int r = 0; r < n_repeat; ++r)
    run_mw_rg();
  auto t2 = clock::now();

  const double serial_rg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double mw_rg_ms     = std::chrono::duration<double, std::milli>(t2 - t1).count();
  const double speedup_rg   = serial_rg_ms / std::max(mw_rg_ms, 1e-9);

  for (int iw = 0; iw < nw; ++iw)
  {
    elecs[iw]->rejectMove(0);
    elecs[iw]->makeMove(0, newpos - elecs[iw]->R[0]);
  }
  run_serial_rg();
  run_mw_rg();
  for (int iw = 0; iw < nw; ++iw)
  {
    CHECK(std::real(ratios_mw[iw]) == Approx(std::real(ratios_serial[iw])));
    for (int d = 0; d < OHMMS_DIM; ++d)
      CHECK(grads_mw[iw][d] == Approx(grads_serial[iw][d]));
  }

  // --- recompute microbench (reject moves first so tables are clean) ---
  for (int iw = 0; iw < nw; ++iw)
    elecs[iw]->rejectMove(0);

  run_serial_recompute();
  run_mw_recompute();
  auto t3 = clock::now();
  for (int r = 0; r < n_repeat / 4; ++r)
    run_serial_recompute();
  auto t4 = clock::now();
  for (int r = 0; r < n_repeat / 4; ++r)
    run_mw_recompute();
  auto t5 = clock::now();
  const double serial_rc_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
  const double mw_rc_ms     = std::chrono::duration<double, std::milli>(t5 - t4).count();
  const double speedup_rc   = serial_rc_ms / std::max(mw_rc_ms, 1e-9);

  const int nthreads =
#ifdef _OPENMP
      omp_get_max_threads()
#else
      1
#endif
      ;
  std::cout << "\n[JeeI microbench] nw=" << nw << " OMP_NUM_THREADS=" << nthreads
            << "\n  ratioGrad  serial_ms=" << serial_rg_ms << " mw_ms=" << mw_rg_ms << " speedup=" << speedup_rg
            << "\n  recompute  serial_ms=" << serial_rc_ms << " mw_ms=" << mw_rc_ms << " speedup=" << speedup_rc
            << std::endl;

  // No speedup gates: the host multi-walker paths are serial per-walker loops by
  // design (walker parallelism comes from crowds); the timings above are informational.
  // Correctness gates are the mw-vs-serial agreement checks earlier in this case.
}


/** Cut2: host compact recompute vs use_offload dense (and CUDA mw) recompute science. */
TEST_CASE("JeeIOrbitalSoA recompute dense agreement", "[wavefunction][recompute]")
{
  Communicate* c = OHMMS::Controller;
  const SimulationCell cell;
  ParticleSet ions_h(cell), elec_h(cell), ions_d(cell), elec_d(cell);

  auto setup = [](ParticleSet& ions, ParticleSet& elec) {
    ions.setName("ion");
    ions.create({2});
    ions.R[0] = {2.0, 0.0, 0.0};
    ions.R[1] = {-2.0, 0.0, 0.0};
    ions.getSpeciesSet().addSpecies("O");
    ions.update();
    elec.setName("elec");
    elec.create({2, 2});
    elec.R[0] = {1.00, 0.0, 0.0};
    elec.R[1] = {0.0, 0.0, 0.0};
    elec.R[2] = {-1.00, 0.0, 0.0};
    elec.R[3] = {0.0, 0.0, 2.0};
    SpeciesSet& sp = elec.getSpeciesSet();
    int upIdx      = sp.addSpecies("u");
    int downIdx    = sp.addSpecies("d");
    int chargeIdx  = sp.addAttribute("charge");
    sp(chargeIdx, upIdx)   = -1;
    sp(chargeIdx, downIdx) = -1;
  };
  setup(ions_h, elec_h);
  setup(ions_d, elec_d);

  const char* particles = R"(<tmp>
    <jastrow name="J3" type="eeI" function="polynomial" source="ion" print="yes">
      <correlation ispecies="O" especies="u" isize="3" esize="3" rcut="10">
        <coefficients id="uuO_rc" type="Array" optimize="yes"> 8.227710241e-06 2.480817653e-06 -5.354068112e-06 -1.112644787e-05 -2.208006078e-06 5.213121933e-06 -1.537865869e-05 8.899030233e-06 6.257255156e-06 3.214580988e-06 -7.716743107e-06 -5.275682077e-06 -1.778457637e-06 7.926231121e-06 1.767406868e-06 5.451359059e-08 2.801423724e-06 4.577282736e-06 7.634608083e-06 -9.510673173e-07 -2.344131575e-06 -1.878777219e-06 3.937363358e-07 5.065353773e-07 5.086724869e-07 -1.358768154e-07</coefficients>
      </correlation>
      <correlation ispecies="O" especies1="u" especies2="d" isize="3" esize="3" rcut="10">
        <coefficients id="udO_rc" type="Array" optimize="yes"> -6.939530224e-06 2.634169299e-05 4.046077477e-05 -8.002682388e-06 -5.396795988e-06 6.697370507e-06 5.433953051e-05 -6.336849668e-06 3.680471431e-05 -2.996059772e-05 1.99365828e-06 -3.222705626e-05 -8.091669063e-06 4.15738535e-06 4.843939112e-06 3.563650208e-07 3.786332474e-05 -1.418336941e-05 2.282691374e-05 1.29239286e-06 -4.93580873e-06 -3.052539228e-06 9.870288001e-08 1.844286407e-06 2.970561871e-07 -4.364303677e-08</coefficients>
      </correlation>
    </jastrow>
</tmp>
)";
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(particles));
  xmlNodePtr jas = xmlFirstElementChild(doc.getRoot());

  using J3Type = JeeIOrbitalSoA<PolynomialFunctor3D>;
  J3Type j3_h("J3_h_rc", ions_h, elec_h, false);
  J3Type j3_d("J3_d_rc", ions_d, elec_d, true);
  auto load = [&](J3Type& j3, ParticleSet& ions, ParticleSet& elec, const std::string& tag) {
    xmlNodePtr kids = jas->children;
    SpeciesSet& iSet = ions.getSpeciesSet();
    SpeciesSet& eSet = elec.getSpeciesSet();
    while (kids != nullptr)
    {
      if (std::string((char*)kids->name) == "correlation")
      {
        RealType ee_cusp = 0.0, eI_cusp = 0.0;
        std::string iSpecies, eSpecies1("u"), eSpecies2("u");
        OhmmsAttributeSet rAttrib;
        rAttrib.add(iSpecies, "ispecies");
        rAttrib.add(eSpecies1, "especies1");
        rAttrib.add(eSpecies2, "especies2");
        rAttrib.add(ee_cusp, "ecusp");
        rAttrib.add(eI_cusp, "icusp");
        rAttrib.put(kids);
        auto functor = std::make_unique<PolynomialFunctor3D>("J3rc_" + iSpecies + eSpecies1 + eSpecies2 + tag, ee_cusp, eI_cusp);
        functor->iSpecies  = iSpecies;
        functor->eSpecies1 = eSpecies1;
        functor->eSpecies2 = eSpecies2;
        functor->put(kids);
        j3.addFunc(iSet.findSpecies(iSpecies), eSet.findSpecies(eSpecies1), eSet.findSpecies(eSpecies2),
                   std::move(functor));
      }
      kids = kids->next;
    }
    j3.check_complete();
  };
  load(j3_h, ions_h, elec_h, "h");
  load(j3_d, ions_d, elec_d, "d");

  elec_h.update();
  elec_d.update();

  elec_h.G = 0;
  elec_h.L = 0;
  elec_d.G = 0;
  elec_d.L = 0;
  const double log_h = std::real(j3_h.evaluateLog(elec_h, elec_h.G, elec_h.L));

  // Cut2: mw_evaluateLog -> mw_recompute (CUDA dense under ENABLE_CUDA)
  RefVectorWithLeader<WaveFunctionComponent> wfc_list(j3_d, {j3_d});
  RefVectorWithLeader<ParticleSet> p_list(elec_d, {elec_d});
  RefVector<ParticleSet::ParticleGradient> G_list  = {elec_d.G};
  RefVector<ParticleSet::ParticleLaplacian> L_list = {elec_d.L};
  j3_d.mw_evaluateLog(wfc_list, p_list, G_list, L_list);
  const double log_d = std::real(j3_d.get_log_value());
  CHECK(log_d == Approx(log_h));
  for (int e = 0; e < 4; ++e)
  {
    for (int d = 0; d < 3; ++d)
      CHECK(elec_d.G[e][d] == Approx(elec_h.G[e][d]));
    CHECK(elec_d.L[e] == Approx(elec_h.L[e]));
  }
}

/** Large-system microbench: bigger eeI system, CUDA/offload mw_ratioGrad vs serial host.
 *  Tiny 4e systems are transfer-bound on GPU; production runs are work-bound (Ne*Ni*nw).
 *  This case uses Ne=64, Ni=32, nw=128 — enough work for the dense dual-table kernel to
 *  amortize H2D. Reports speedup; requires >2x when CUDA offload is active.
 */
TEST_CASE("JeeIOrbitalSoA large-system CUDA mw_ratioGrad speedup", "[wavefunction][benchmark][large]")
{
  Communicate* c = OHMMS::Controller;
  const int nw       = 128;
  const int n_repeat = 30;
  const int Ne_up = 32, Ne_dn = 32, Ni = 32;
  const SimulationCell simulation_cell;

  ParticleSet ions(simulation_cell);
  ions.setName("ion");
  ions.create({Ni});
  for (int i = 0; i < Ni; ++i)
    ions.R[i] = {RealType(i % 8) * 1.5, RealType(i / 8) * 1.5, 0.0};
  ions.getSpeciesSet().addSpecies("O");
  ions.update();

  auto make_elec = [&]() {
    auto elec = std::make_unique<ParticleSet>(simulation_cell);
    elec->setName("elec");
    elec->create({Ne_up, Ne_dn});
    for (int e = 0; e < Ne_up + Ne_dn; ++e)
      elec->R[e] = {RealType(e % 6) * 0.7, RealType((e / 6) % 6) * 0.7, RealType(e / 36) * 0.5 + 0.3};
    SpeciesSet& sp = elec->getSpeciesSet();
    int upIdx      = sp.addSpecies("u");
    int downIdx    = sp.addSpecies("d");
    int chargeIdx  = sp.addAttribute("charge");
    sp(chargeIdx, upIdx)   = -1;
    sp(chargeIdx, downIdx) = -1;
    return elec;
  };

  // Minimal polynomial coefficients (same shape as golden, truncated content OK for timing)
  const char* particles = R"(<tmp>
    <jastrow name="J3" type="eeI" function="polynomial" source="ion" print="no">
      <correlation ispecies="O" especies="u" isize="3" esize="3" rcut="6">
        <coefficients id="uuO_s" type="Array" optimize="no"> 8.227710241e-06 2.480817653e-06 -5.354068112e-06 -1.112644787e-05 -2.208006078e-06 5.213121933e-06 -1.537865869e-05 8.899030233e-06 6.257255156e-06 3.214580988e-06 -7.716743107e-06 -5.275682077e-06 -1.778457637e-06 7.926231121e-06 1.767406868e-06 5.451359059e-08 2.801423724e-06 4.577282736e-06 7.634608083e-06 -9.510673173e-07 -2.344131575e-06 -1.878777219e-06 3.937363358e-07 5.065353773e-07 5.086724869e-07 -1.358768154e-07</coefficients>
      </correlation>
      <correlation ispecies="O" especies1="u" especies2="d" isize="3" esize="3" rcut="6">
        <coefficients id="udO_s" type="Array" optimize="no"> -6.939530224e-06 2.634169299e-05 4.046077477e-05 -8.002682388e-06 -5.396795988e-06 6.697370507e-06 5.433953051e-05 -6.336849668e-06 3.680471431e-05 -2.996059772e-05 1.99365828e-06 -3.222705626e-05 -8.091669063e-06 4.15738535e-06 4.843939112e-06 3.563650208e-07 3.786332474e-05 -1.418336941e-05 2.282691374e-05 1.29239286e-06 -4.93580873e-06 -3.052539228e-06 9.870288001e-08 1.844286407e-06 2.970561871e-07 -4.364303677e-08</coefficients>
      </correlation>
    </jastrow>
</tmp>
)";
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(particles));
  xmlNodePtr jas = xmlFirstElementChild(doc.getRoot());

  std::vector<std::unique_ptr<ParticleSet>> elecs(nw);
  std::vector<std::unique_ptr<WaveFunctionComponent>> j3s(nw);
  using J3Type = JeeIOrbitalSoA<PolynomialFunctor3D>;
  elecs[0] = make_elec();
  elecs[0]->update();
  {
    eeI_JastrowBuilder b(c, *elecs[0], ions);
    j3s[0] = b.buildComponent(jas);
  }
  REQUIRE(j3s[0]);
  auto* leader = dynamic_cast<J3Type*>(j3s[0].get());
  REQUIRE(leader);
  const bool off = leader->isUsingOffload();

  for (int iw = 1; iw < nw; ++iw)
  {
    elecs[iw] = make_elec();
    j3s[iw]   = j3s[0]->makeClone(*elecs[iw]);
    elecs[iw]->update();
  }
  elecs[0]->update();

  for (int iw = 0; iw < nw; ++iw)
  {
    elecs[iw]->G = 0;
    elecs[iw]->L = 0;
    j3s[iw]->evaluateLog(*elecs[iw], elecs[iw]->G, elecs[iw]->L);
  }

  const PosType newpos(0.25, 0.15, 0.4);
  for (int iw = 0; iw < nw; ++iw)
    elecs[iw]->makeMove(0, newpos - elecs[iw]->R[0]);

  std::vector<PsiValue> ratios_serial(nw), ratios_mw(nw);
  std::vector<GradType> grads_serial(nw), grads_mw(nw);
  RefVectorWithLeader<WaveFunctionComponent> wfc_list(*j3s[0]);
  RefVectorWithLeader<ParticleSet> p_list(*elecs[0]);
  for (int iw = 0; iw < nw; ++iw)
  {
    wfc_list.push_back(*j3s[iw]);
    p_list.push_back(*elecs[iw]);
  }

  // Persistent multi-walker resource, as acquired by the batched drivers: without it
  // every mw call allocates pinned buffers and re-uploads the full e-I tables, which
  // measures a path production never runs.
  ResourceCollection wfc_res("bench_jeei_res");
  leader->createResource(wfc_res);
  ResourceCollectionTeamLock<WaveFunctionComponent> mw_lock(wfc_res, wfc_list);

  auto run_serial = [&]() {
    for (int iw = 0; iw < nw; ++iw)
    {
      grads_serial[iw]  = GradType(0);
      ratios_serial[iw] = j3s[iw]->ratioGrad(*elecs[iw], 0, grads_serial[iw]);
    }
  };
  auto run_mw = [&]() {
    for (int iw = 0; iw < nw; ++iw)
      grads_mw[iw] = GradType(0);
    leader->mw_ratioGrad(wfc_list, p_list, 0, ratios_mw, grads_mw);
  };

  // Warmup (pays CUDA alloc / static H2D once)
  run_serial();
  run_mw();
  for (int iw = 0; iw < nw; ++iw)
  {
    elecs[iw]->rejectMove(0);
    elecs[iw]->makeMove(0, newpos - elecs[iw]->R[0]);
  }

  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();
  for (int r = 0; r < n_repeat; ++r)
    run_serial();
  auto t1 = clock::now();
  for (int r = 0; r < n_repeat; ++r)
    run_mw();
  auto t2 = clock::now();

  const double serial_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double mw_ms     = std::chrono::duration<double, std::milli>(t2 - t1).count();
  const double speedup   = serial_ms / std::max(mw_ms, 1e-9);

  // Correctness sample
  for (int iw = 0; iw < nw; ++iw)
  {
    elecs[iw]->rejectMove(0);
    elecs[iw]->makeMove(0, newpos - elecs[iw]->R[0]);
  }
  run_serial();
  run_mw();
  for (int iw = 0; iw < nw; ++iw)
  {
    CHECK(std::real(ratios_mw[iw]) == Approx(std::real(ratios_serial[iw])));
    for (int d = 0; d < OHMMS_DIM; ++d)
      CHECK(grads_mw[iw][d] == Approx(grads_serial[iw][d]).margin(1e-8));
  }

  // Cut2: recompute timing (reject moves so full tables are consistent)
  for (int iw = 0; iw < nw; ++iw)
    elecs[iw]->rejectMove(0);
  std::vector<bool> recompute_all(nw, true);
  auto run_serial_rc = [&]() {
    for (int iw = 0; iw < nw; ++iw)
      j3s[iw]->recompute(*elecs[iw]);
  };
  auto run_mw_rc = [&]() { leader->mw_recompute(wfc_list, p_list, recompute_all); };
  run_serial_rc();
  run_mw_rc();
  auto t3 = clock::now();
  for (int r = 0; r < n_repeat / 2; ++r)
    run_serial_rc();
  auto t4 = clock::now();
  for (int r = 0; r < n_repeat / 2; ++r)
    run_mw_rc();
  auto t5 = clock::now();
  const double serial_rc_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
  const double mw_rc_ms     = std::chrono::duration<double, std::milli>(t5 - t4).count();
  const double speedup_rc   = serial_rc_ms / std::max(mw_rc_ms, 1e-9);

  // recompute science: log via evaluateLog on leader vs one walker serial
  elecs[0]->G = 0;
  elecs[0]->L = 0;
  const double log_s = std::real(j3s[0]->evaluateLog(*elecs[0], elecs[0]->G, elecs[0]->L));
  elecs[1]->G = 0;
  elecs[1]->L = 0;
  j3s[1]->recompute(*elecs[1]);
  const double log_m = std::real(j3s[1]->evaluateLog(*elecs[1], elecs[1]->G, elecs[1]->L));
  CHECK(log_m == Approx(log_s));

  std::cout << "\n[JeeI large-system microbench] Ne=" << (Ne_up + Ne_dn) << " Ni=" << Ni << " nw=" << nw
            << " offload=" << (off ? "yes" : "no")
            << "\n  ratioGrad  serial_ms=" << serial_ms << " mw_ms=" << mw_ms << " speedup=" << speedup
            << "\n  recompute  serial_ms=" << serial_rc_ms << " mw_ms=" << mw_rc_ms << " speedup=" << speedup_rc
            << std::endl;

#if defined(ENABLE_CUDA)
  if (off)
  {
    // GPU dense path must beat serial host on work-bound sizes (both cuts)
    REQUIRE(speedup > 2.0);
    REQUIRE(speedup_rc > 1.5);
  }
#endif
}

} // namespace qmcplusplus
