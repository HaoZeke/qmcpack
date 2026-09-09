//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: Rohit Goswami, rgoswami@ieee.org, SURF
//
// File created by: Rohit Goswami, rgoswami@ieee.org, SURF
//////////////////////////////////////////////////////////////////////////////////////

#include "MultiBsplineOffloadMapperFactory.hpp"
#if defined(HAVE_MPI)
#include "MultiBsplineMPIShared.hpp"
#include "MultiBsplineOffloadMapperPeer.hpp"
#endif

namespace qmcplusplus
{
template<typename T>
std::shared_ptr<MultiBsplineOffloadMapper<T>> makeOffloadMapper(const MultiBsplineBase<T>& host_bsplines)
{
#if defined(HAVE_MPI)
  if (const auto* shared = dynamic_cast<const MultiBsplineMPIShared<T>*>(&host_bsplines))
    if (shared->getSharingComm().size() > 1)
      return std::make_shared<MultiBsplineOffloadMapperPeer<T>>(*shared, shared->getSharingComm());
#endif
  return std::make_shared<MultiBsplineOffloadMapper<T>>(host_bsplines);
}

template std::shared_ptr<MultiBsplineOffloadMapper<float>> makeOffloadMapper(const MultiBsplineBase<float>&);
template std::shared_ptr<MultiBsplineOffloadMapper<double>> makeOffloadMapper(const MultiBsplineBase<double>&);
} // namespace qmcplusplus
