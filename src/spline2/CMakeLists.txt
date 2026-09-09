#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2021 QMCPACK developers.
#//
#// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//
#// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////


add_library(spline2_omptarget OBJECT MultiBsplineOffloadMapper.cpp)
if(HAVE_MPI)
  # one device copy of the coefficients shared across the ranks of a node, and the
  # factory that chooses it. Both need the offload compile options this target carries
  # and MPI, so they cannot live in the plain spline2 target. message carries the
  # include path for the bundled mpi3 wrapper that Message/Communicate.h pulls in.
  # The peer mapper needs a device runtime with inter-process memory handles, and
  # compiles to the plain per-rank mapping without one.
  target_sources(spline2_omptarget PRIVATE MultiBsplineOffloadMapperPeer.cpp
                                           MultiBsplineOffloadMapperFactory.cpp)
  target_link_libraries(spline2_omptarget PRIVATE message)
endif()
target_link_libraries(spline2_omptarget PUBLIC einspline platform_runtime)

set(SPLINE2_SRC MultiBspline.cpp MultiBspline1D.cpp SplineUtils.cpp)
if(HAVE_MPI)
  list(APPEND SPLINE2_SRC MultiBsplineMPIShared.cpp)
endif()

if(USE_OBJECT_TARGET)
  add_library(spline2 OBJECT ${SPLINE2_SRC})
else()
  add_library(spline2 ${SPLINE2_SRC})
endif()
target_link_libraries(spline2 PUBLIC spline2_omptarget message qmcio)

if(BUILD_UNIT_TESTS)
  add_subdirectory(tests)
endif()
