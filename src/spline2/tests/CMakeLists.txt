#//////////////////////////////////////////////////////////////////////////////////////
#// This file is distributed under the University of Illinois/NCSA Open Source License.
#// See LICENSE file in top directory for details.
#//
#// Copyright (c) 2018 Jeongnim Kim and QMCPACK developers.
#//
#// File developed by:  Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
#//
#// File created by: Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
#//////////////////////////////////////////////////////////////////////////////////////

set(SRC_DIR spline2)
set(UTEST_EXE test_${SRC_DIR})
set(UTEST_NAME deterministic-unit_test_${SRC_DIR})

set(SRCS test_multi_spline.cpp)
if(HAVE_MPI)
  list(APPEND SRCS test_multi_spline_MPI_shared.cpp)
endif()

add_executable(${UTEST_EXE} ${SRCS})
if(HAVE_MPI)
  target_link_libraries(${UTEST_EXE} message)
endif()
target_link_libraries(${UTEST_EXE} catch_main spline2 einspline qmcutil)
if(USE_OBJECT_TARGET)
  target_link_libraries(${UTEST_EXE} spline2_omptarget)
endif()

add_unit_test(${UTEST_NAME} 1 1 $<TARGET_FILE:${UTEST_EXE}>)
if(HAVE_MPI)
  add_unit_test(${UTEST_NAME}-r2 2 1 $<TARGET_FILE:${UTEST_EXE}> "[shared-offload]")
endif()

# Compares mapped-device VGH+trace and direct VGL evaluation across orbital counts.
if(ENABLE_OFFLOAD)
  add_executable(benchmark_vgl_offload EXCLUDE_FROM_ALL benchmark_vgl_offload.cpp)
  target_link_libraries(benchmark_vgl_offload spline2 einspline qmcutil platform_runtime)
  if(USE_OBJECT_TARGET)
    target_link_libraries(benchmark_vgl_offload spline2_omptarget)
  endif()
endif()
