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


#include "MultiBsplineMPISharedOffload.hpp"

namespace qmcplusplus
{
template class MultiBsplineMPISharedOffload<float>;
template class MultiBsplineMPISharedOffload<double>;
} // namespace qmcplusplus
