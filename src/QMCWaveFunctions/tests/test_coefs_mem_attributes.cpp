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

#include <catch2/catch_test_macros.hpp>

#include "OhmmsData/Libxml2Doc.h"
#include "QMCWaveFunctions/BsplineFactory/EinsplineSetBuilder.h"
#include "Message/Communicate.h"

#include <string>

namespace qmcplusplus
{
namespace testing
{
std::pair<int, int> obtainMemoryAttributes(xmlNodePtr cur) { return EinsplineSetBuilder::obtainMemoryAttributes(cur); }
} // namespace testing

/// the <sposet> node obtainMemoryAttributes reads, with whatever coefs_mem it is given
static std::string sposetWith(const std::string& coefs_mem)
{
  return R"(<sposet type="bspline" href="dummy.h5" size="4">)" + coefs_mem + "</sposet>";
}

TEST_CASE("coefs_mem attributes", "[wavefunction]")
{
  SECTION("defaults are one and one")
  {
    Libxml2Document doc;
    REQUIRE(doc.parseFromString(sposetWith("")));
    const auto [distributed_ranks, shared_ranks] = testing::obtainMemoryAttributes(doc.getRoot());
    CHECK(distributed_ranks == 1);
    CHECK(shared_ranks == 1);
  }

  SECTION("both are read from the node")
  {
    Libxml2Document doc;
    REQUIRE(doc.parseFromString(sposetWith(R"(<coefs_mem distributed_ranks="2" shared_ranks="3"/>)")));
    const auto [distributed_ranks, shared_ranks] = testing::obtainMemoryAttributes(doc.getRoot());
    CHECK(distributed_ranks == 2);
    CHECK(shared_ranks == 3);
  }

  SECTION("values below one are raised to one")
  {
    Libxml2Document doc;
    REQUIRE(doc.parseFromString(sposetWith(R"(<coefs_mem distributed_ranks="0" shared_ranks="-4"/>)")));
    const auto [distributed_ranks, shared_ranks] = testing::obtainMemoryAttributes(doc.getRoot());
    CHECK(distributed_ranks == 1);
    CHECK(shared_ranks == 1);
  }

  SECTION("a product the ranks per node do not divide is refused")
  {
    const int node_comm_size = OHMMS::Controller->NodeComm().size();

    // shared_ranks alone has to be able to fail the check. The product is what the
    // ranks per node must divide, so a request the node cannot honour is refused
    // whichever of the two attributes carries it.
    Libxml2Document shared_only;
    REQUIRE(shared_only.parseFromString(sposetWith(R"(<coefs_mem shared_ranks="7"/>)")));
    if (node_comm_size % 7 > 0)
      CHECK_THROWS_AS(testing::obtainMemoryAttributes(shared_only.getRoot()), std::runtime_error);
    else
      CHECK_NOTHROW(testing::obtainMemoryAttributes(shared_only.getRoot()));

    // and so does the product of two values each of which divides it
    Libxml2Document both;
    REQUIRE(both.parseFromString(sposetWith(R"(<coefs_mem distributed_ranks="1" shared_ranks="5"/>)")));
    if (node_comm_size % 5 > 0)
      CHECK_THROWS_AS(testing::obtainMemoryAttributes(both.getRoot()), std::runtime_error);
    else
      CHECK_NOTHROW(testing::obtainMemoryAttributes(both.getRoot()));
  }
}
} // namespace qmcplusplus
