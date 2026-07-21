//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


#ifndef OHMMS_OPENMP_H
#define OHMMS_OPENMP_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#else
using omp_int_t = int;
inline omp_int_t omp_get_thread_num() { return 0; }
inline omp_int_t omp_get_max_threads() { return 1; }
inline omp_int_t omp_get_num_threads() { return 1; }
inline omp_int_t omp_get_level() { return 0; }
inline omp_int_t omp_get_ancestor_thread_num(int level) { return 0; }
inline omp_int_t omp_get_max_active_levels() { return 1; }
inline void omp_set_num_threads(int num_threads) {}
#endif

/** Threads available for a nested parallel region.
 *  Must not open a parallel region only to count threads: that is a full barrier
 *  and is on the delayed-update flush path every time delay fills (norb >= 256).
 *  When already inside a parallel region (batched DMC crowd threads), return 1 so
 *  det GEMM / DT / Coulomb do not oversubscribe via nested OpenMP.
 */
inline int getNextLevelNumThreads()
{
  if (omp_get_level() > 0)
    return 1;
  return omp_get_max_threads();
}

#endif // OHMMS_COMMUNICATE_H
