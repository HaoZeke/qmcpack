//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: Rohit Goswami, rohit.goswami@surf.nl, SURF
//
// File created by: Rohit Goswami, rohit.goswami@surf.nl, SURF
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>

#include "spline2/MultiBsplineMPIShared.hpp"
#include "spline2/SplineUtils.h"
#include "einspline/bspline_create.h"
#include "hdf/hdf_archive.h"
#include "Message/Communicate.h"
#include "Utilities/FairDivide.h"

#include <filesystem>
#include <vector>

namespace qmcplusplus
{
namespace
{
/// a smooth field to fill the coefficients with, so a block that came back wrong is visible
double sample_field(int i, int j, int k, int n)
{
  return std::sin(2 * M_PI * i / n) * std::cos(2 * M_PI * j / n) + 0.25 * k;
}

template<typename T>
std::unique_ptr<MultiBsplineMPIShared<T>> makeTable(const Ugrid grid[3],
                                                    const BCtype_d bc[3],
                                                    size_t num_splines,
                                                    unsigned distributed_ranks)
{
  // Communicate's second argument is how many groups to split into, so one
  // group of every rank is what holds a table distributed over all of them.
  // The table's blocks live in one MPI-3 shared window, so those ranks are on
  // one node either way.
  auto comm = std::make_unique<Communicate>(*OHMMS::Controller, OHMMS::Controller->size() / distributed_ranks);
  return std::make_unique<MultiBsplineMPIShared<T>>(grid, bc, num_splines, std::move(comm), distributed_ranks);
}
} // namespace

/** A spline dump holds one dataset per coefficient block, named for that block.
 *
 * The number of blocks is the number of ranks the table is distributed over, so
 * a run reads back a dump written under a different distribution and has to be
 * able to tell. Two things make that possible and both are checked here: every
 * block's dataset carries its own index in its name, and the dump records how
 * many blocks it holds.
 */
TEST_CASE("spline dump names one dataset per block", "[spline2]")
{
  Communicate* comm                = OHMMS::Controller;
  const unsigned distributed_ranks = comm->size();

  const int n = 8;
  Ugrid grid[3];
  BCtype_d bc[3];
  for (int d = 0; d < 3; d++)
  {
    grid[d].start = 0.0;
    grid[d].end   = 1.0;
    grid[d].num   = n;
    bc[d].lCode   = PERIODIC;
    bc[d].rCode   = PERIODIC;
    bc[d].lVal    = 0.0;
    bc[d].rVal    = 0.0;
  }

  std::vector<double> data(n * n * n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        data[(i * n + j) * n + k] = sample_field(i, j, k, n);

  const size_t num_splines = 12;
  auto written             = makeTable<double>(grid, bc, num_splines, distributed_ranks);
  REQUIRE(written->getNumBlocks() == distributed_ranks);

  UBspline_3d_d* one = create_UBspline_3d_d(grid[0], grid[1], grid[2], bc[0], bc[1], bc[2], data.data());
  auto offsets       = FairDivideAligned<std::vector<size_t>>(num_splines, getAlignment<double>(), comm->size());
  for (size_t i = offsets[comm->rank()]; i < offsets[comm->rank() + 1]; i++)
    written->set_spline(*one, i);
  destroy_Bspline(one);
  comm->barrier();

  const std::filesystem::path file("spline_dump_per_block.h5");
  int wrote = 0;
  if (comm->rank() == 0)
  {
    hdf_archive h5f;
    if (h5f.create(file))
    {
      int num_blocks = written->getNumBlocks();
      h5f.write(num_blocks, splineDumpNumBlocksName());
      wrote = SplineUtils<double>::write(*written, h5f);
      h5f.close();
    }
  }
  comm->bcast(wrote);
  REQUIRE(wrote == 1);

  auto restored = makeTable<double>(grid, bc, num_splines, distributed_ranks);

  // Rank 0 does the reading, and every rank asserts the outcome. A failing
  // assertion inside a rank guard ends that rank's test case and leaves the
  // others at the next barrier, so what crosses the guard is the answers.
  int names_ok = 0, no_extra_dataset = 0, count_ok = 0, read_ok = 0, coefs_ok = 0;
  if (comm->rank() == 0)
  {
    hdf_archive h5f;
    if (h5f.open(file, H5F_ACC_RDONLY))
    {
      // Naming each block for its own index is what lets a dump be read back one
      // block at a time. A name built by appending to the previous one
      // round-trips through this very code, and is visible only from outside it,
      // so it is the names in the file that are checked and not just the values.
      names_ok = 1;
      for (size_t iblock = 0; iblock < written->getNumBlocks(); iblock++)
        if (!h5f.is_dataset(blockDatasetName(iblock)))
          names_ok = 0;
      no_extra_dataset = !h5f.is_dataset(blockDatasetName(written->getNumBlocks()));

      int num_blocks = 0;
      count_ok = h5f.readEntry(num_blocks, splineDumpNumBlocksName()) &&
          static_cast<size_t>(num_blocks) == written->getNumBlocks();

      read_ok = SplineUtils<double>::read(*restored, h5f);
      h5f.close();
    }

    if (read_ok)
    {
      coefs_ok = 1;
      for (size_t iblock = 0; iblock < written->getNumBlocks(); iblock++)
      {
        const auto& from = written->getBlock(iblock);
        const auto& to   = restored->getBlock(iblock);
        if (to.coefs_size != from.coefs_size)
        {
          coefs_ok = 0;
          continue;
        }
        for (size_t i = 0; i < from.coefs_size; i++)
          if (to.coefs[i] != from.coefs[i])
          {
            coefs_ok = 0;
            break;
          }
      }
    }

    std::filesystem::remove(file);
  }

  comm->bcast(names_ok);
  comm->bcast(no_extra_dataset);
  comm->bcast(count_ok);
  comm->bcast(read_ok);
  comm->bcast(coefs_ok);

  CHECK(names_ok == 1);
  CHECK(no_extra_dataset == 1);
  CHECK(count_ok == 1);
  CHECK(read_ok == 1);
  CHECK(coefs_ok == 1);
}

} // namespace qmcplusplus
