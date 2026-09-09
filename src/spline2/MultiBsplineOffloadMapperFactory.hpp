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
// -*- C++ -*-
/**@file MultiBsplineOffloadMapperFactory.hpp
 *
 * choose a device mapper for a host spline table
 */
#ifndef QMCPLUSPLUS_MULTIEINSPLINE_OFFLOAD_MAPPER_FACTORY_HPP
#define QMCPLUSPLUS_MULTIEINSPLINE_OFFLOAD_MAPPER_FACTORY_HPP

#include <memory>
#include "MultiBsplineOffloadMapper.hpp"

namespace qmcplusplus
{
/** the device mapper a host table wants.
 *
 * A table whose coefficients sit in a shared host window is read by a group of ranks
 * that can also share one device copy, which is what limits walkers per device. Any
 * other table gets one device copy per rank. The choice is made from the table's own
 * type, so an SPO class asks for a mapper without knowing how the table is held.
 */
template<typename T>
std::shared_ptr<MultiBsplineOffloadMapper<T>> makeOffloadMapper(const MultiBsplineBase<T>& host_bsplines);

extern template std::shared_ptr<MultiBsplineOffloadMapper<float>> makeOffloadMapper(const MultiBsplineBase<float>&);
extern template std::shared_ptr<MultiBsplineOffloadMapper<double>> makeOffloadMapper(const MultiBsplineBase<double>&);
} // namespace qmcplusplus

#endif
