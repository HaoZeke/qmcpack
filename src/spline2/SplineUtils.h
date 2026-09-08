//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_SPLINE_UTILS_H
#define QMCPLUSPLUS_SPLINE_UTILS_H

#include "hdf/hdf_archive.h"
#include "Message/Communicate.h"

#include <string>

namespace qmcplusplus
{
template<typename T>
class MultiBsplineBase;
template<typename T>
class MultiBspline1D;

/** name of the dataset holding one block of coefficients in a spline dump
 *
 * A table distributed over N ranks is N blocks, each its own dataset, so the
 * name has to carry the block index. Save and restore share this one function
 * so a dump can only be read back under the name it was written with.
 */
inline std::string blockDatasetName(size_t iblock) { return "spline_" + std::to_string(iblock); }

/// number of coefficient blocks a dump holds, recorded so a restore can refuse a different distribution
inline const char* splineDumpNumBlocksName() { return "num_blocks"; }

template<typename ST>
class SplineUtils
{
public:
  static bool read(MultiBsplineBase<ST>& spline, hdf_archive& h5f);
  static bool write(MultiBsplineBase<ST>& spline, hdf_archive& h5f);

  static bool read(MultiBspline1D<ST>& spline, hdf_archive& h5f);
  static bool write(MultiBspline1D<ST>& spline, hdf_archive& h5f);

  static void gatherv(MultiBsplineBase<ST>& spline, size_t iblock, const std::vector<int>& offset, Communicate& comm);
  static void bcast(MultiBsplineBase<ST>& spline, size_t iblock, Communicate& comm);

  static void gatherv(MultiBspline1D<ST>& spline, size_t stride, const std::vector<int>& offset, Communicate& comm);
  static void bcast(MultiBspline1D<ST>& spline, Communicate& comm);
};

extern template class SplineUtils<float>;
extern template class SplineUtils<double>;


} // namespace qmcplusplus
#endif
