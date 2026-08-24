//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Raymond Clay III, j.k.rofling@gmail.com, Lawrence Livermore National Laboratory
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jaron T. Krogel, krogeljt@ornl.gov, Oak Ridge National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#include "WaveFunctionComponent.h"
#include <omp.h>

namespace qmcplusplus
{
// for return types
using PsiValue = WaveFunctionComponent::PsiValue;

WaveFunctionComponent::WaveFunctionComponent(const std::string& obj_name)
    : UpdateMode(ORB_WALKER), Bytes_in_WFBuffer(0), my_name_(obj_name), log_value_(0.0)
{}

WaveFunctionComponent::~WaveFunctionComponent() = default;

void WaveFunctionComponent::mw_evaluateLog(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                           const RefVectorWithLeader<ParticleSet>& p_list,
                                           const RefVector<ParticleSet::ParticleGradient>& G_list,
                                           const RefVector<ParticleSet::ParticleLaplacian>& L_list) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    wfc_list[iw].evaluateLog(p_list[iw], G_list[iw], L_list[iw]);
}

void WaveFunctionComponent::recompute(const ParticleSet& P)
{
  ParticleSet::ParticleGradient temp_G(P.getTotalNum());
  ParticleSet::ParticleLaplacian temp_L(P.getTotalNum());

  evaluateLog(P, temp_G, temp_L);
}

void WaveFunctionComponent::mw_recompute(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                         const RefVectorWithLeader<ParticleSet>& p_list,
                                         const std::vector<bool>& recompute) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    if (recompute[iw])
      wfc_list[iw].recompute(p_list[iw]);
}

void WaveFunctionComponent::mw_prepareGroup(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                            const RefVectorWithLeader<ParticleSet>& p_list,
                                            int ig) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    wfc_list[iw].prepareGroup(p_list[iw], ig);
}

template<CoordsType CT>
void WaveFunctionComponent::mw_evalGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                        const RefVectorWithLeader<ParticleSet>& p_list,
                                        const int iat,
                                        TWFGrads<CT>& grad_now) const
{
  if constexpr (CT == CoordsType::POS_SPIN)
    mw_evalGradWithSpin(wfc_list, p_list, iat, grad_now.grads_positions, grad_now.grads_spins);
  else
    mw_evalGrad(wfc_list, p_list, iat, grad_now.grads_positions);
}

void WaveFunctionComponent::mw_evalGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                        const RefVectorWithLeader<ParticleSet>& p_list,
                                        int iat,
                                        std::vector<GradType>& grad_now) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    grad_now[iw] = wfc_list[iw].evalGrad(p_list[iw], iat);
}

void WaveFunctionComponent::mw_evalGradDevice(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                              const RefVectorWithLeader<ParticleSet>& p_list,
                                              int iat,
                                              std::vector<GradType>& grad_now,
                                              Vector<ValueType, OffloadPinnedAllocator<ValueType>>& grads_device_now,
                                              bool assign) const
{
  const int nw = wfc_list.size();
  mw_evalGrad(wfc_list, p_list, iat, grad_now);

  /* The host values have to reach the sum the device side steps read, so they are staged
   * once and added there. A component that overrides this never forms them at all.
   */
  constexpr int dim = OHMMS_DIM;
  Vector<ValueType, OffloadPinnedAllocator<ValueType>> staged(nw * dim);
  for (int iw = 0; iw < nw; iw++)
    for (int id = 0; id < dim; id++)
      staged[iw * dim + id] = static_cast<ValueType>(grad_now[iw][id]);
  staged.updateTo();

  const auto* src_ptr = staged.device_data();
  auto* dst_ptr       = grads_device_now.device_data();
  PRAGMA_OFFLOAD("omp target teams distribute parallel for is_device_ptr(src_ptr, dst_ptr)")
  for (int iw = 0; iw < nw; iw++)
    for (int id = 0; id < dim; id++)
      if (assign)
        dst_ptr[iw * dim + id] = src_ptr[iw * dim + id];
      else
        dst_ptr[iw * dim + id] += src_ptr[iw * dim + id];
}

void WaveFunctionComponent::mw_evalGradWithSpin(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                                const RefVectorWithLeader<ParticleSet>& p_list,
                                                int iat,
                                                std::vector<GradType>& grad_now,
                                                std::vector<ComplexType>& spingrad_now) const
{
  mw_evalGrad(wfc_list, p_list, iat, grad_now);
  for (int iw = 0; iw < wfc_list.size(); iw++)
    spingrad_now[iw] = 0;
}

void WaveFunctionComponent::mw_evalGradWithSpin_serialized(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                                           const RefVectorWithLeader<ParticleSet>& p_list,
                                                           int iat,
                                                           std::vector<GradType>& grad_now,
                                                           std::vector<ComplexType>& spingrad_now) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
  {
    spingrad_now[iw] = 0;
    grad_now[iw]     = wfc_list[iw].evalGradWithSpin(p_list[iw], iat, spingrad_now[iw]);
  }
}

void WaveFunctionComponent::mw_calcRatio(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                         const RefVectorWithLeader<ParticleSet>& p_list,
                                         int iat,
                                         std::vector<PsiValue>& ratios) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    ratios[iw] = wfc_list[iw].ratio(p_list[iw], iat);
}


PsiValue WaveFunctionComponent::ratioGrad(ParticleSet& P, int iat, GradType& grad_iat)
{
  APP_ABORT("WaveFunctionComponent::ratioGrad is not implemented in " + getClassName() + " class.");
  return ValueType();
}

template<CoordsType CT>
void WaveFunctionComponent::mw_ratioGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                         const RefVectorWithLeader<ParticleSet>& p_list,
                                         int iat,
                                         std::vector<PsiValue>& ratios,
                                         TWFGrads<CT>& grad_new) const
{
  if constexpr (CT == CoordsType::POS_SPIN)
    mw_ratioGradWithSpin(wfc_list, p_list, iat, ratios, grad_new.grads_positions, grad_new.grads_spins);
  else
    mw_ratioGrad(wfc_list, p_list, iat, ratios, grad_new.grads_positions);
}

void WaveFunctionComponent::mw_ratioGrad(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                         const RefVectorWithLeader<ParticleSet>& p_list,
                                         int iat,
                                         std::vector<PsiValue>& ratios,
                                         std::vector<GradType>& grad_new) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    ratios[iw] = wfc_list[iw].ratioGrad(p_list[iw], iat, grad_new[iw]);
}

void WaveFunctionComponent::mw_accept_rejectMoveFromDeviceMask(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int iat,
    const char* accept_mask,
    const std::vector<bool>& isAccepted,
    bool safe_to_delay) const
{
  /* A component without its own device path still has to accept, and the decision it needs is
   * already on the host: the caller fetched it once for all of them. Fetching it here instead
   * is a blocking transfer per component per electron for the same nw bytes.
   */
  mw_accept_rejectMove(wfc_list, p_list, iat, isAccepted, safe_to_delay);
}

void WaveFunctionComponent::mw_ratioGradDevice(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                               const RefVectorWithLeader<ParticleSet>& p_list,
                                               int iat,
                                               std::vector<PsiValue>& ratios,
                                               std::vector<GradType>& grad_new,
                                               Vector<PsiValue, OffloadPinnedAllocator<PsiValue>>& ratios_device_prod,
                                               Vector<ValueType, OffloadPinnedAllocator<ValueType>>& grads_device_sum,
                                               bool assign) const
{
  // the component's own contribution, before the caller's running values are touched
  const int nw = wfc_list.size();
  std::vector<GradType> grad_z(nw, GradType(0));
  mw_ratioGrad(wfc_list, p_list, iat, ratios, grad_z);

  // a component without a device path still owes the caller its factor and its term, so
  // the host results go down and are folded in there
  constexpr int dim = OHMMS_DIM;
  host_ratio_staging_.resize(nw);
  host_grad_staging_.resize(nw * dim);
  for (int iw = 0; iw < nw; iw++)
  {
    host_ratio_staging_[iw] = ratios[iw];
    for (int id = 0; id < dim; id++)
      host_grad_staging_[iw * dim + id] = grad_z[iw][id];
  }
  host_ratio_staging_.updateTo();
  host_grad_staging_.updateTo();

  const auto* z_ptr = host_ratio_staging_.device_data();
  const auto* g_ptr = host_grad_staging_.device_data();
  auto* prod_ptr    = ratios_device_prod.device_data();
  auto* gsum_ptr    = grads_device_sum.device_data();
  PRAGMA_OFFLOAD("omp target teams distribute parallel for is_device_ptr(z_ptr, g_ptr, prod_ptr, gsum_ptr)")
  for (int iw = 0; iw < nw; iw++)
  {
    if (assign)
    {
      prod_ptr[iw] = z_ptr[iw];
      for (int id = 0; id < dim; id++)
        gsum_ptr[iw * dim + id] = g_ptr[iw * dim + id];
    }
    else
    {
      prod_ptr[iw] *= z_ptr[iw];
      for (int id = 0; id < dim; id++)
        gsum_ptr[iw * dim + id] += g_ptr[iw * dim + id];
    }
  }

  // the caller's host gradient accumulates the same way the non device form does
  for (int iw = 0; iw < nw; iw++)
    grad_new[iw] += grad_z[iw];
}

void WaveFunctionComponent::mw_ratioGradWithSpin(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                                 const RefVectorWithLeader<ParticleSet>& p_list,
                                                 int iat,
                                                 std::vector<PsiValue>& ratios,
                                                 std::vector<GradType>& grad_new,
                                                 std::vector<ComplexType>& spingrad_new) const
{ mw_ratioGrad(wfc_list, p_list, iat, ratios, grad_new); }

void WaveFunctionComponent::mw_ratioGradWithSpin_serialized(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                                            const RefVectorWithLeader<ParticleSet>& p_list,
                                                            int iat,
                                                            std::vector<PsiValue>& ratios,
                                                            std::vector<GradType>& grad_new,
                                                            std::vector<ComplexType>& spingrad_new) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    ratios[iw] = wfc_list[iw].ratioGradWithSpin(p_list[iw], iat, grad_new[iw], spingrad_new[iw]);
}

void WaveFunctionComponent::mw_accept_rejectMove(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                                 const RefVectorWithLeader<ParticleSet>& p_list,
                                                 int iat,
                                                 const std::vector<bool>& isAccepted,
                                                 bool safe_to_delay) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    if (isAccepted[iw])
      wfc_list[iw].acceptMove(p_list[iw], iat, safe_to_delay);
    else
      wfc_list[iw].restore(iat);
}

void WaveFunctionComponent::mw_completeUpdates(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    wfc_list[iw].completeUpdates();
}

WaveFunctionComponent::LogValue WaveFunctionComponent::evaluateGL(const ParticleSet& P,
                                                                  ParticleSet::ParticleGradient& G,
                                                                  ParticleSet::ParticleLaplacian& L,
                                                                  bool fromscratch)
{ return evaluateLog(P, G, L); }

void WaveFunctionComponent::mw_evaluateGL(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                          const RefVectorWithLeader<ParticleSet>& p_list,
                                          const RefVector<ParticleSet::ParticleGradient>& G_list,
                                          const RefVector<ParticleSet::ParticleLaplacian>& L_list,
                                          bool fromscratch) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
    wfc_list[iw].evaluateGL(p_list[iw], G_list[iw], L_list[iw], fromscratch);
}

void WaveFunctionComponent::extractOptimizableObjectRefs(UniqueOptObjRefs&)
{
  if (isOptimizable())
    throw std::logic_error("Bug!! " + getClassName() +
                           "::extractOptimizableObjectRefs "
                           "must be overloaded when the WFC is optimizable.");
}

void WaveFunctionComponent::checkOutVariables(const OptVariables& active)
{
  if (isOptimizable())
    throw std::logic_error("Bug!! " + getClassName() +
                           "::checkOutVariables "
                           "must be overloaded when the WFC is optimizable.");
}

void WaveFunctionComponent::evaluateDerivativesWF(ParticleSet& P,
                                                  const OptVariables& active,
                                                  Vector<ValueType>& dlogpsi)
{ throw std::runtime_error("WaveFunctionComponent::evaluateDerivativesWF is not implemented by " + getClassName()); }


/*@todo makeClone should be a pure virtual function
 */
std::unique_ptr<WaveFunctionComponent> WaveFunctionComponent::makeClone(ParticleSet& tpq) const
{
  APP_ABORT("Implement WaveFunctionComponent::makeClone " + getClassName() + " class.");
  return std::unique_ptr<WaveFunctionComponent>();
}

WaveFunctionComponent::RealType WaveFunctionComponent::KECorrection() { return 0; }

void WaveFunctionComponent::evaluateRatiosAlltoOne(ParticleSet& P, std::vector<ValueType>& ratios)
{
  assert(P.getTotalNum() == ratios.size());
  for (int i = 0; i < P.getTotalNum(); ++i)
    ratios[i] = ratio(P, i);
}

void WaveFunctionComponent::evaluateRatios(const VirtualParticleSet& P, std::vector<ValueType>& ratios)
{
  std::ostringstream o;
  o << "WaveFunctionComponent::evaluateRatios is not implemented by " << getClassName();
  APP_ABORT(o.str());
}

void WaveFunctionComponent::evaluateSpinorRatios(const VirtualParticleSet& P,
                                                 const std::pair<ValueVector, ValueVector>& spinor_multiplier,
                                                 std::vector<ValueType>& ratios)
{ evaluateRatios(P, ratios); }

void WaveFunctionComponent::mw_evaluateRatios(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                                              const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
                                              std::vector<std::vector<ValueType>>& ratios) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
  {
    /* Serialising to the single-walker call carries one reference electron per set, so a
     * set spanning several would take the first one's ratio for every quadrature point.
     * supportsMultiRefRatios() exists so callers can avoid building such a set; refuse it
     * here as well, because the wrong answer is otherwise silent.
     */
    if (vp_list[iw].isMultiRef())
      throw std::runtime_error(getClassName() +
                               " has no multi-walker ratio path, so it evaluates one reference electron per "
                               "virtual particle set. It was handed a set spanning several.");
    wfc_list[iw].evaluateRatios(vp_list[iw], ratios[iw]);
  }
}

void WaveFunctionComponent::mw_evaluateSpinorRatios(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
    const RefVector<std::pair<ValueVector, ValueVector>>& spinor_multiplier_list,
    std::vector<std::vector<ValueType>>& ratios) const
{ mw_evaluateRatios(wfc_list, vp_list, ratios); }

void WaveFunctionComponent::mw_evaluateSpinorRatios_serialized(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
    const RefVector<std::pair<ValueVector, ValueVector>>& spinor_multiplier_list,
    std::vector<std::vector<ValueType>>& ratios) const
{
  assert(this == &wfc_list.getLeader());
  for (int iw = 0; iw < wfc_list.size(); iw++)
  {
    // one reference electron per set, as in mw_evaluateRatios above
    if (vp_list[iw].isMultiRef())
      throw std::runtime_error(getClassName() +
                               " evaluates spinor ratios one walker at a time, so it takes one reference "
                               "electron per virtual particle set. It was handed a set spanning several.");
    wfc_list[iw].evaluateSpinorRatios(vp_list[iw], spinor_multiplier_list[iw], ratios[iw]);
  }
}

void WaveFunctionComponent::evaluateDerivRatios(const VirtualParticleSet& VP,
                                                const OptVariables& optvars,
                                                std::vector<ValueType>& ratios,
                                                Matrix<ValueType>& dratios)
{
  //default is only ratios and zero derivatives
  evaluateRatios(VP, ratios);
}

void WaveFunctionComponent::evaluateSpinorDerivRatios(const VirtualParticleSet& VP,
                                                      const std::pair<ValueVector, ValueVector>& spinor_multiplier,
                                                      const OptVariables& optvars,
                                                      std::vector<ValueType>& ratios,
                                                      Matrix<ValueType>& dratios)
{ evaluateDerivRatios(VP, optvars, ratios, dratios); }

void WaveFunctionComponent::registerTWFFastDerivWrapper(const ParticleSet& P, TWFFastDerivWrapper& twf) const
{
  std::ostringstream o;
  o << "WaveFunctionComponent::registerTWFFastDerivWrapper is not implemented by " << getClassName();
  APP_ABORT(o.str());
}

template void WaveFunctionComponent::mw_evalGrad<CoordsType::POS>(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int iat,
    TWFGrads<CoordsType::POS>& grad_now) const;
template void WaveFunctionComponent::mw_evalGrad<CoordsType::POS_SPIN>(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int iat,
    TWFGrads<CoordsType::POS_SPIN>& grad_now) const;
template void WaveFunctionComponent::mw_ratioGrad<CoordsType::POS>(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int iat,
    std::vector<PsiValue>& ratios,
    TWFGrads<CoordsType::POS>& grad_new) const;
template void WaveFunctionComponent::mw_ratioGrad<CoordsType::POS_SPIN>(
    const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
    const RefVectorWithLeader<ParticleSet>& p_list,
    int iat,
    std::vector<PsiValue>& ratios,
    TWFGrads<CoordsType::POS_SPIN>& grad_new) const;

} // namespace qmcplusplus
