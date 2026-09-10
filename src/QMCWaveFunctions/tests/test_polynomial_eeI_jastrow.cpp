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
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "OhmmsData/Libxml2Doc.h"
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

using std::string;

namespace qmcplusplus
{
using RealType = WaveFunctionComponent::RealType;
using LogValue = WaveFunctionComponent::LogValue;
using PsiValue = WaveFunctionComponent::PsiValue;

TEST_CASE("PolynomialFunctor3D functor zero", "[wavefunction]")
{
  PolynomialFunctor3D functor("test_functor");

  double r = 1.2;
  double u = functor.evaluate(r, r, r);
  REQUIRE(u == 0.0);
}

TEST_CASE("PolynomialFunctor3D device value gradient and hessian", "[wavefunction]")
{
  using real_type = PolynomialFunctor3D::real_type;

  PolynomialFunctor3D functor("test_functor");
  functor.cutoff_radius = 2.5;
  functor.resize(3, 3);
  for (size_t i = 0; i < functor.Parameters.size(); i++)
    functor.Parameters[i] = 0.017 * static_cast<real_type>((i % 5) + 1) - 0.023 * static_cast<real_type>((i % 3) + 1);
  functor.reset_gamma();

  std::vector<real_type> gamma_flat(functor.gammaFlatSize());
  functor.copyGammaFlat(gamma_flat.data());
  const real_type L = 0.5 * functor.cutoff_radius;

  /* The device form carries no screening, because the compression that feeds it keeps
   * only triplets inside the ion cutoff. Sampling outside would compare against the zero
   * the host returns there.
   */
  const real_type rs[] = {0.2, 0.6, 1.0};
  for (real_type r_12 : rs)
    for (real_type r_1I : rs)
      for (real_type r_2I : rs)
      {
        TinyVector<real_type, 3> grad;
        Tensor<real_type, 3> hess;
        const real_type val_host = functor.evaluate(r_12, r_1I, r_2I, grad, hess);

        real_type v, g0, g1, g2, h00, h01, h02, h11, h22;
        PolynomialFunctor3D::evaluateVGH_impl(r_12, r_1I, r_2I, gamma_flat.data(), functor.N_eI, functor.N_ee,
                                              functor.C, L, v, g0, g1, g2, h00, h01, h02, h11, h22);

        CHECK(v == Approx(val_host));
        // the device form leaves each derivative divided by its distances
        CHECK(g0 * r_12 == Approx(grad[0]));
        CHECK(g1 * r_1I == Approx(grad[1]));
        CHECK(g2 * r_2I == Approx(grad[2]));
        CHECK(h00 == Approx(hess(0, 0)));
        CHECK(h11 == Approx(hess(1, 1)));
        CHECK(h22 == Approx(hess(2, 2)));
        CHECK(h01 * (r_12 * r_1I) == Approx(hess(0, 1)));
        CHECK(h02 * (r_12 * r_2I) == Approx(hess(0, 2)));
      }
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

/** the batched accept has to leave the same state the single walker accept does.
 *
 * JeeIOrbitalSoA keeps Uat, dUat and d2Uat across moves and an accept updates them
 * incrementally, so log_value_ after an accept is a running total. Recomputing the
 * term from the accepted configuration has to give the same number. If it does not,
 * every evaluation after the accept is made against a wavefunction that has drifted,
 * which is what a shift in the pseudopotential energy looks like from the outside.
 *
 * No reference numbers here on purpose. The incremental state is checked against a
 * recompute of the same configuration, so the test cannot pass by agreeing with a
 * figure that was itself wrong.
 */
void test_J3_batched_accept(const DynamicCoordinateKind kind_selected)
{
  const SimulationCell simulation_cell;
  ParticleSet ions_(simulation_cell, kind_selected);
  ions_.setName("ion");
  ions_.create({2});
  ions_.R[0] = {2.0, 0.0, 0.0};
  ions_.R[1] = {-2.0, 0.0, 0.0};
  SpeciesSet& source_species(ions_.getSpeciesSet());
  source_species.addSpecies("C");
  ions_.update();

  // two walkers, so the batched accept has more than one move to place on the device
  const int nw = 2;
  std::vector<ParticleSet> elecs(nw, ParticleSet(simulation_cell, kind_selected));
  for (int iw = 0; iw < nw; iw++)
  {
    auto& e = elecs[iw];
    e.setName("elec");
    e.create({2, 2});
    e.R[0] = {1.00, 0.0, 0.0};
    e.R[1] = {0.0, 0.0, 0.0};
    e.R[2] = {-1.00, 0.0, 0.0};
    e.R[3] = {0.0, 0.0, 2.0};
    // the two walkers are not the same configuration, or a per walker indexing
    // fault would cancel
    e.R[1][1] += 0.1 * (iw + 1);
    e.R[3][2] -= 0.05 * (iw + 1);
    SpeciesSet& target_species(e.getSpeciesSet());
    const int upIdx                    = target_species.addSpecies("u");
    const int downIdx                  = target_species.addSpecies("d");
    const int chargeIdx                = target_species.addAttribute("charge");
    target_species(chargeIdx, upIdx)   = -1;
    target_species(chargeIdx, downIdx) = -1;
  }

  const char* particles = R"(<tmp>
  <jastrow name="J3" type="eeI" function="polynomial" source="ion" print="yes">
     <correlation ispecies="C" especies1="u" especies2="u" isize="3" esize="3" rcut="10">
       <coefficients id="uuC" type="Array" optimize="yes"> 8.227710241e-06 2.480817653e-06 -5.354068112e-06 -1.112644787e-05 -2.208006078e-06 5.213121933e-06 -1.537865869e-05 8.899030233e-06 6.257255036e-06 3.214580988e-06 -7.716743051e-06 -5.275682993e-06 -1.778457637e-06 7.926492166e-06 1.767718682e-06 5.451693643e-06 5.099229044e-06 -7.259144046e-06 -3.331387256e-06 -1.917186297e-06 -1.352371308e-06 3.395795298e-06 -7.207717903e-06 -8.417589123e-06</coefficients>
     </correlation>
     <correlation ispecies="C" especies1="u" especies2="d" isize="3" esize="3" rcut="10">
       <coefficients id="udC" type="Array" optimize="yes"> -6.939530224e-06 2.634169299e-05 4.046256499e-06 6.397449517e-06 2.116502065e-06 -1.482828074e-05 -2.017346198e-06 -1.834722340e-06 -2.128333939e-06 6.677160687e-06 7.484349260e-06 -1.180825291e-06 -1.617132404e-06 -2.607480350e-06 -1.365385358e-06 -3.006045849e-06 -6.454295212e-06 -1.293972790e-06 -1.397631976e-06 -1.482828074e-06 -1.834722340e-06 -2.017346198e-06 -2.128333939e-06 6.677160687e-06</coefficients>
     </correlation>
  </jastrow>
</tmp>)";
  Libxml2Document doc;
  REQUIRE(doc.parseFromString(particles));
  xmlNodePtr jas_eeI = xmlFirstElementChild(doc.getRoot());

  Communicate* c = OHMMS::Controller;
  using J3Type   = JeeIOrbitalSoA<PolynomialFunctor3D>;
  std::vector<std::unique_ptr<WaveFunctionComponent>> j3s;
  for (int iw = 0; iw < nw; iw++)
  {
    eeI_JastrowBuilder jastrow(c, elecs[iw], ions_);
    j3s.push_back(jastrow.buildComponent(jas_eeI));
    REQUIRE(dynamic_cast<J3Type*>(j3s[iw].get()) != nullptr);
    elecs[iw].update();
  }

  RefVector<ParticleSet> p_refs;
  RefVector<WaveFunctionComponent> j_refs;
  for (int iw = 0; iw < nw; iw++)
  {
    p_refs.push_back(elecs[iw]);
    j_refs.push_back(*j3s[iw]);
  }
  RefVectorWithLeader<ParticleSet> p_list(elecs[0], p_refs);
  RefVectorWithLeader<WaveFunctionComponent> j_list(*j3s[0], j_refs);

  ResourceCollection pset_res("test_pset");
  ResourceCollection wfc_res("test_wfc");
  elecs[0].createResource(pset_res);
  j3s[0]->createResource(wfc_res);
  ResourceCollectionTeamLock<ParticleSet> mw_pset_lock(pset_res, p_list);
  ResourceCollectionTeamLock<WaveFunctionComponent> mw_wfc_lock(wfc_res, j_list);

  std::vector<ParticleSet::ParticleGradient> G(nw, ParticleSet::ParticleGradient(4));
  std::vector<ParticleSet::ParticleLaplacian> L(nw, ParticleSet::ParticleLaplacian(4));
  for (int iw = 0; iw < nw; iw++)
    j3s[iw]->evaluateLog(elecs[iw], G[iw], L[iw]);

  /* Two accepted moves, not one. The device path asks the electron-ion table for
   * its temporary distances through requireTempDataOnDevice, and a table only
   * starts producing them on the move after the request, so the first accept takes
   * the fallback whatever the deck says. One move would test the fallback twice.
   */
  const int moved_elec_id = 1;
  for (int step = 0; step < 2; step++)
  {
    std::vector<ParticleSet::SingleParticlePos> displs(nw);
    for (int iw = 0; iw < nw; iw++)
      displs[iw] = {0.1 + 0.01 * iw, -0.05, 0.02 * (step + 1)};

    ParticleSet::mw_makeMove(p_list, moved_elec_id, displs);
    std::vector<PsiValue> ratios(nw);
    j3s[0]->mw_calcRatio(j_list, p_list, moved_elec_id, ratios);

    std::vector<bool> isAccepted(nw, true);
    j3s[0]->mw_accept_rejectMove(j_list, p_list, moved_elec_id, isAccepted, false);
    ParticleSet::mw_accept_rejectMove(p_list, moved_elec_id, isAccepted, true);
  }

  // what the accepts left behind, against a recompute of the configuration they left
  for (int iw = 0; iw < nw; iw++)
  {
    /* evaluateGL forms the gradient and the laplacian from the stored Uat, dUat and
     * d2Uat, so it reports what the accepts maintained. evaluateLog rebuilds those
     * from the configuration. The two have to agree, and the order matters: the
     * recompute overwrites the state, so the incremental figures are taken first.
     */
    ParticleSet::ParticleGradient G_inc(4);
    ParticleSet::ParticleLaplacian L_inc(4);
    const LogValue incremental = j3s[iw]->evaluateGL(elecs[iw], G_inc, L_inc, false);

    ParticleSet::ParticleGradient G_fresh(4);
    ParticleSet::ParticleLaplacian L_fresh(4);
    elecs[iw].update();
    const LogValue recomputed = j3s[iw]->evaluateLog(elecs[iw], G_fresh, L_fresh);

    CHECK(std::real(incremental) == Approx(std::real(recomputed)));
    for (int iel = 0; iel < 4; iel++)
    {
      CHECK(std::real(L_inc[iel]) == Approx(std::real(L_fresh[iel])));
      for (int idim = 0; idim < OHMMS_DIM; idim++)
        CHECK(std::real(G_inc[iel][idim]) == Approx(std::real(G_fresh[iel][idim])));
    }
  }
}

TEST_CASE("PolynomialFunctor3D Jastrow", "[wavefunction]")
{
  test_J3_polynomial3D(DynamicCoordinateKind::DC_POS);
  test_J3_polynomial3D(DynamicCoordinateKind::DC_POS_OFFLOAD);
}

TEST_CASE("eeI Jastrow batched accept against a recompute", "[wavefunction]")
{
  test_J3_batched_accept(DynamicCoordinateKind::DC_POS);
  test_J3_batched_accept(DynamicCoordinateKind::DC_POS_OFFLOAD);
}
} // namespace qmcplusplus
