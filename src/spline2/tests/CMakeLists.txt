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
  # The shared and divided cases need a group of ranks: at one rank a sharing group
  # shares with itself and every one of them returns early, so the same executable
  # also runs on two and on four. The assertion counts differ between them, which is
  # the point of registering more than one.
  add_unit_test(${UTEST_NAME}-r2 2 1 $<TARGET_FILE:${UTEST_EXE}>)
  add_unit_test(${UTEST_NAME}-r4 4 1 $<TARGET_FILE:${UTEST_EXE}>)
endif()
