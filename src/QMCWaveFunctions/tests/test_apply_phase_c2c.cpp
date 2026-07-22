//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>

#include "QMCWaveFunctions/BsplineFactory/contraction_helper.hpp"
#include "QMCWaveFunctions/BsplineFactory/ApplyPhaseC2C.hpp"

#include <algorithm>
#include <complex>
#include <vector>

namespace qmcplusplus
{
TEST_CASE("C2C scalar phase VGL spans a spline team boundary", "[wavefunction]")
{
  using ST = float;
  using TT = std::complex<double>;

  constexpr size_t num_orbitals       = 257;
  constexpr size_t spline_padded_size = num_orbitals * 2;
  constexpr size_t chunk_size         = 512;
  constexpr size_t num_teams          = (spline_padded_size + chunk_size - 1) / chunk_size;

  std::vector<ST> spline_vgl(spline_padded_size * SoAFields3D::NUM_FIELDS);
  for (size_t field = 0; field < SoAFields3D::NUM_FIELDS; ++field)
    for (size_t index = 0; index < spline_padded_size; ++index)
      spline_vgl[field * spline_padded_size + index] =
          ST(0.003 * static_cast<double>(field + 1) + 0.0002 * static_cast<double>(index + 1));

  std::vector<ST> kcart(num_orbitals * 3);
  std::vector<ST> mkk(num_orbitals);
  for (size_t index = 0; index < num_orbitals; ++index)
  {
    kcart[index]                    = ST(0.01 + 0.0001 * index);
    kcart[num_orbitals + index]     = ST(-0.02 + 0.0002 * index);
    kcart[num_orbitals * 2 + index] = ST(0.03 - 0.00015 * index);
    mkk[index] = -(kcart[index] * kcart[index] + kcart[num_orbitals + index] * kcart[num_orbitals + index] +
                   kcart[num_orbitals * 2 + index] * kcart[num_orbitals * 2 + index]);
  }

  const ST G[9]  = {ST(1.1), ST(0.2), ST(-0.1), ST(0.05), ST(0.9), ST(0.3), ST(-0.2), ST(0.1), ST(1.2)};
  constexpr ST x = ST(0.25);
  constexpr ST y = ST(-0.5);
  constexpr ST z = ST(0.75);

  std::vector<TT> expected(num_orbitals * 5);
  std::vector<TT> actual(num_orbitals * 5);
  std::vector<TT> expected_value(num_orbitals);
  std::vector<TT> actual_value(num_orbitals);

  for (size_t team_id = 0; team_id < num_teams; ++team_id)
  {
    const size_t first = chunk_size * team_id;
    const size_t last  = std::min(first + chunk_size, spline_padded_size);
    size_t first_complex;
    size_t last_complex;
    C2C::complex_index_bounds(first, last, num_orbitals, first_complex, last_complex);

    if (team_id == 0)
    {
      CHECK(first_complex == 0);
      CHECK(last_complex == 256);
    }
    else
    {
      CHECK(first_complex == 256);
      CHECK(last_complex == 257);
    }

    for (size_t index = first_complex; index < last_complex; ++index)
    {
      C2C::assign_v(x, y, z, expected_value.data(), spline_vgl.data(), kcart.data(), num_orbitals, index);

      const size_t jr = index * 2;
      const size_t ji = jr + 1;
      actual_value[index] =
          C2C::apply_phase_value<ST, TT>(x, y, z, spline_vgl[jr], spline_vgl[ji], kcart[index],
                                         kcart[num_orbitals + index], kcart[num_orbitals * 2 + index]);

      C2C::assign_vgl(x, y, z, expected.data(), num_orbitals, mkk.data(), spline_vgl.data(), spline_padded_size, G,
                      kcart.data(), num_orbitals, index);

      C2C::apply_phase_vgl(x, y, z, spline_vgl[SoAFields3D::VAL * spline_padded_size + jr],
                           spline_vgl[SoAFields3D::VAL * spline_padded_size + ji],
                           spline_vgl[SoAFields3D::GRAD0 * spline_padded_size + jr],
                           spline_vgl[SoAFields3D::GRAD0 * spline_padded_size + ji],
                           spline_vgl[SoAFields3D::GRAD1 * spline_padded_size + jr],
                           spline_vgl[SoAFields3D::GRAD1 * spline_padded_size + ji],
                           spline_vgl[SoAFields3D::GRAD2 * spline_padded_size + jr],
                           spline_vgl[SoAFields3D::GRAD2 * spline_padded_size + ji],
                           spline_vgl[SoAFields3D::LAPL * spline_padded_size + jr],
                           spline_vgl[SoAFields3D::LAPL * spline_padded_size + ji], G, kcart[index],
                           kcart[num_orbitals + index], kcart[num_orbitals * 2 + index], mkk[index], actual[index],
                           actual[num_orbitals + index], actual[num_orbitals * 2 + index],
                           actual[num_orbitals * 3 + index], actual[num_orbitals * 4 + index]);
    }
  }

  for (size_t index = 0; index < num_orbitals; ++index)
    CHECK(actual_value[index] == expected_value[index]);

  for (size_t field = 0; field < 5; ++field)
    for (size_t index = 0; index < num_orbitals; ++index)
      CHECK(actual[field * num_orbitals + index] == expected[field * num_orbitals + index]);
}
} // namespace qmcplusplus
