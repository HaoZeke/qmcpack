//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_DRIFTMODIFIER_UNR_H
#define QMCPLUSPLUS_DRIFTMODIFIER_UNR_H

#include <cmath>
#include <limits>

#include "QMCDrivers/GreenFunctionModifiers/DriftModifierBase.h"
#include "config.h"

namespace qmcplusplus
{
/** Umrigar drift scaling, callable from a target region.
 *
 *  The scaling depends only on tau, the parameter a and the squared drift, so it
 *  is plain arithmetic on scalars. getDrift is virtual and a virtual call cannot
 *  cross into a target region, which is what keeps the DMC acceptance test on
 *  the host; the arithmetic itself has no such restriction and is shared here so
 *  both sides compute the same value.
 */
PRAGMA_OFFLOAD("omp begin declare target")
template<typename T>
inline T driftScalingUNR(T tau, T a, T vsq)
{
  return vsq < std::numeric_limits<T>::epsilon() ? tau : ((T(-1) + std::sqrt(T(1) + T(2) * a * tau * vsq)) / (a * vsq));
}
PRAGMA_OFFLOAD("omp end declare target")

class DriftModifierUNR : public DriftModifierBase
{
public:
  using RealType = QMCTraits::RealType;
  using PosType  = QMCTraits::PosType;

  void getDrifts(RealType tau, const std::vector<GradType>& qf, std::vector<PosType>&) const final;

  void getDrift(RealType tau, const GradType& qf, PosType& drift) const final;

  void getDrifts(RealType tau,
                 const std::vector<ComplexType>& qf,
                 std::vector<ParticleSet::Scalar_t>& drift) const final;

  void getDrift(RealType tau, const ComplexType& qf, ParticleSet::Scalar_t& drift) const final;

  bool parseXML(xmlNodePtr cur) final;

  /** the drifts for one electron across the walkers, written to device memory
   *
   *  getDrift is virtual and cannot be called from a target region; the scaling it applies
   *  is shared through driftScalingUNR so this computes the same values. grads and drifts
   *  are device pointers: the gradient is summed there and the proposal is formed there,
   *  so the drift never appears on the host. Both are [nw][DIM]
   *  flat as the SPOSet device entry point reports them, and drifts is written in the same
   *  layout, so a device side move can read it without the host forming the drift.
   */
  void getDriftsDevice(RealType tau,
                       size_t nw,
                       int dim,
                       const QMCTraits::ValueType* grads,
                       RealType* drifts) const override
  {
    const RealType a = a_;
    PRAGMA_OFFLOAD("omp target teams distribute parallel for is_device_ptr(grads, drifts)")
    for (size_t iw = 0; iw < nw; iw++)
    {
      RealType vsq(0);
      for (int idim = 0; idim < dim; idim++)
      {
        const RealType g = std::real(grads[iw * dim + idim]);
        vsq += g * g;
      }
      const RealType sc = driftScalingUNR(tau, a, vsq);
      for (int idim = 0; idim < dim; idim++)
        drifts[iw * dim + idim] = std::real(grads[iw * dim + idim]) * sc;
    }
  }

  bool isUNRScaling() const override { return true; }
  RealType getUNRScalingA() const override { return a_; }

  DriftModifierUNR(RealType a = 1.0) : a_(a) {}

private:
  /// JCP1993 Umrigar et eq. (35) "a" parameter is set to 1.0
  RealType a_;
};

} // namespace qmcplusplus

#endif
