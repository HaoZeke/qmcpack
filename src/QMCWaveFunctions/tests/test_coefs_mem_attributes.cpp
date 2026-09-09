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
#include <catch2/matchers/catch_matchers_string.hpp>

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
  // Every request is checked against the ranks per node, so what a section may ask for
  // depends on how many ranks are running it. A value of 1 divides any count, and any
  // other value is asserted against this.
  const int ranks_per_node = OHMMS::Controller->NodeComm().size();

  SECTION("defaults are one and one")
  {
    Libxml2Document doc;
    REQUIRE(doc.parseFromString(sposetWith("")));
    const auto [distributed_ranks, shared_ranks] = testing::obtainMemoryAttributes(doc.getRoot());
    CHECK(distributed_ranks == 1);
    CHECK(shared_ranks == 1);
  }

  SECTION("values below one are raised to one")
  {
    Libxml2Document doc;
    REQUIRE(doc.parseFromString(sposetWith(R"(<coefs_mem distributed_ranks="0" shared_ranks="-4"/>)")));
    const auto [distributed_ranks, shared_ranks] = testing::obtainMemoryAttributes(doc.getRoot());
    CHECK(distributed_ranks == 1);
    CHECK(shared_ranks == 1);
  }

  SECTION("both attributes are read, and both take part in the check")
  {
    Libxml2Document doc;
    REQUIRE(doc.parseFromString(sposetWith(R"(<coefs_mem distributed_ranks="2" shared_ranks="3"/>)")));
    if (ranks_per_node % 6 == 0)
    {
      const auto [distributed_ranks, shared_ranks] = testing::obtainMemoryAttributes(doc.getRoot());
      CHECK(distributed_ranks == 2);
      CHECK(shared_ranks == 3);
    }
    else
    {
      // the message names the product, so it says what both attributes were read as
      CHECK_THROWS_WITH(testing::obtainMemoryAttributes(doc.getRoot()),
                        Catch::Matchers::ContainsSubstring("distributed_ranks and shared_ranks (6)"));
    }
  }

  SECTION("shared_ranks alone can fail the check")
  {
    // The product is what the ranks per node have to divide, so a request the node
    // cannot honour is refused whichever attribute carries it. With distributed_ranks
    // left at its default, an unparenthesised check has nothing to test.
    Libxml2Document doc;
    REQUIRE(doc.parseFromString(sposetWith(R"(<coefs_mem shared_ranks="7"/>)")));
    if (ranks_per_node % 7 == 0)
    {
      const auto [distributed_ranks, shared_ranks] = testing::obtainMemoryAttributes(doc.getRoot());
      CHECK(distributed_ranks == 1);
      CHECK(shared_ranks == 7);
    }
    else
      CHECK_THROWS_AS(testing::obtainMemoryAttributes(doc.getRoot()), std::runtime_error);
  }
}
} // namespace qmcplusplus
