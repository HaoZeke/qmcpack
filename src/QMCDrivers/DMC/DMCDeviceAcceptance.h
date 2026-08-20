//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_DMC_DEVICE_ACCEPTANCE_H
#define QMCPLUSPLUS_DMC_DEVICE_ACCEPTANCE_H

#include "config.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>

namespace qmcplusplus
{

/** Evaluate the DMC Metropolis predicate for one particle across a walker crowd.
 *
 * All pointer arguments refer to device storage. The validity byte includes position
 * validity and the fixed-node phase predicate. The output uses one byte per walker so
 * it can be consumed directly by target acceptance kernels.
 */
template<typename RT, typename PsiV>
inline void computeDMCDeviceAcceptance(std::size_t num_walkers,
                                       const PsiV* ratios,
                                       const RT* log_gf,
                                       const RT* log_gb,
                                       const char* are_valid,
                                       const RT* variates,
                                       char* accepted)
{
  PRAGMA_OFFLOAD("omp target teams distribute parallel for \
                  is_device_ptr(ratios, log_gf, log_gb, are_valid, variates, accepted)")
  for (std::size_t iw = 0; iw < num_walkers; ++iw)
  {
    const RT probability     = std::norm(ratios[iw]) * std::exp(log_gb[iw] - log_gf[iw]);
    const bool should_accept = are_valid[iw] != 0 && ratios[iw] != PsiV(0) &&
        probability >= std::numeric_limits<RT>::epsilon() && variates[iw] < probability;
    accepted[iw] = should_accept ? 1 : 0;
  }
}

} // namespace qmcplusplus

#endif // QMCPLUSPLUS_DMC_DEVICE_ACCEPTANCE_H
