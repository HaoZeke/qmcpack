//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2018 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//
// File created by: Mark Dewing, mdewing@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "OhmmsSoA/VectorSoaContainer.h"
#include "spline2/MultiBsplineMPIShared.hpp"
#include "spline2/MultiBsplineOffloadMapper.hpp"
#include "spline2/MultiBsplineOffloadMapperPeer.hpp"
#include "spline2/MultiBsplineMPISharedOffload.hpp"
#include "spline2/MultiBsplineEval.hpp"
#include "QMCWaveFunctions/BsplineFactory/contraction_helper.hpp"
#include "config/stdlib/Constants.h"
#include "OMPTarget/OffloadAlignedAllocators.hpp"
#include "einspline/bspline_create.h"

namespace qmcplusplus
{

TEST_CASE("MultiBspline peer topology policy", "[spline2][shared-offload]")
{
  const std::string local_node = "node-a";
  const std::string local_gpu  = "0000:01:00.0";
  const std::vector<std::string> owner_nodes{"node-a", "node-a"};
  const std::vector<std::string> owner_gpus{"0000:01:00.0", "0000:02:00.0"};

  CHECK(detail::localPeerTopologyAllowsSharing(local_node, local_gpu, owner_gpus, owner_nodes, owner_gpus, {1, 1}));
  CHECK_FALSE(
      detail::localPeerTopologyAllowsSharing(local_node, local_gpu, {local_gpu}, owner_nodes, owner_gpus, {1, 1}));
  CHECK_FALSE(detail::localPeerTopologyAllowsSharing(local_node, local_gpu, owner_gpus, {"node-a", "node-b"},
                                                     owner_gpus, {1, 1}));
  CHECK_FALSE(
      detail::localPeerTopologyAllowsSharing(local_node, local_gpu, owner_gpus, owner_nodes, owner_gpus, {1, 0}));
}

TEST_CASE("MultiBspline peer failures are collective", "[spline2][shared-offload][peer-collective]")
{
  auto& comm              = *OHMMS::Controller;
  const auto failure      = detail::collectiveFailure(comm, comm.rank() == 1);
  const bool has_rank_one = comm.size() > 1;
  CHECK(failure.any_failed == has_rank_one);
  CHECK(failure.first_failed_rank == (has_rank_one ? 1 : -1));
}

/** Supports testing many sizes of splines for benchmarking
 *  modified from einspline/tests/test_3d.cpp
 */

template<typename T, int GRID_SIZE>
class test_splines_base
{
protected:
  BCtype_d bc[3];
  Ugrid grid[3];

  const int N = GRID_SIZE;
  double delta;
  std::vector<double> data;

public:
  test_splines_base()
  {
    data.resize(N * N * N);

    grid[0].start = 0.0;
    grid[0].end   = 1.0;
    grid[0].num   = N;

    grid[1].start = 0.0;
    grid[1].end   = 1.0;
    grid[1].num   = N;

    grid[2].start = 0.0;
    grid[2].end   = 1.0;
    grid[2].num   = N;

    delta = (grid[0].end - grid[0].start) / grid[0].num;

    bc[0].lCode = PERIODIC;
    bc[0].rCode = PERIODIC;
    bc[0].lVal  = 0.0;
    bc[0].rVal  = 0.0;
    bc[1].lCode = PERIODIC;
    bc[1].rCode = PERIODIC;
    bc[1].lVal  = 0.0;
    bc[1].rVal  = 0.0;
    bc[2].lCode = PERIODIC;
    bc[2].rCode = PERIODIC;
    bc[2].lVal  = 0.0;
    bc[2].rVal  = 0.0;

    double tpi = 2 * M_PI;
    // Generate the data in double precision regardless of the target spline precision
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
        {
          double x                    = delta * i;
          double y                    = delta * j;
          double z                    = delta * k;
          data[i * N * N + j * N + k] = std::sin(tpi * x) + std::sin(3 * tpi * y) + std::sin(4 * tpi * z);
        }
  }
};

/** Unspecialized test_splines doesn't test eval values
 */
template<typename T, int GRID_SIZE = 5>
struct test_splines : public test_splines_base<T, GRID_SIZE>
{
  using base = test_splines_base<T, GRID_SIZE>;
  using base::bc;
  using base::data;
  using base::delta;
  using base::grid;
  using base::N;

  void test(size_t num_splines)
  {
    // one group spanning every rank: the constructor's communicator covers
    // distributed_ranks * shared_ranks ranks, and this case shares across all of
    // them without distributing, so distributed_ranks is 1
    auto comm_distributed = std::make_unique<Communicate>(*OHMMS::Controller, 1);
    auto& comm(*comm_distributed);
    MultiBsplineMPIShared<T> bs(grid, bc, num_splines, std::move(comm_distributed), 1);

    const size_t npad = getAlignedSize<T>(num_splines);
    REQUIRE(bs.num_splines_padded() == getAlignedSize<T>(num_splines));

    UBspline_3d_d* aspline = create_UBspline_3d_d(grid[0], grid[1], grid[2], bc[0], bc[1], bc[2], data.data());

    auto offsets = FairDivideAligned<std::vector<size_t>>(num_splines, getAlignment<T>(), comm.size());
    for (int i = offsets[comm.rank()]; i < offsets[comm.rank() + 1]; i++)
      bs.set_spline(*aspline, i);
    comm.barrier();

    destroy_Bspline(aspline);

    //  The values for N=5 are not good enough for finer grids so by default we don't do those checks

    TinyVector<T, 3> pos = {0, 0, 0};

    aligned_vector<T> v(npad);
    bs.evaluate_v(pos, v);

    VectorSoaContainer<T, 3> dv(npad);
    VectorSoaContainer<T, 6> hess(npad);
    bs.evaluate_vgh(pos, v, dv, hess);

    pos = {0.1, 0.2, 0.3};
    bs.evaluate_v(pos, v);

    bs.evaluate_vgh(pos, v, dv, hess);

    VectorSoaContainer<T, 3> lap(npad);
    bs.evaluate_vgl(pos, v, dv, lap);

    VectorSoaContainer<T, 10> ghess(npad);
    bs.evaluate_vghgh(pos, v, dv, hess, ghess);
  }
};

/** Partially specialized test_splines does test eval values
 * See gen_bspline_values.py
 */
template<typename T>
struct test_splines<T, 5> : public test_splines_base<T, 5>
{
  using base = test_splines_base<T, 5>;
  using base::bc;
  using base::data;
  using base::delta;
  using base::grid;
  using base::N;

  void test(size_t num_splines, unsigned shared_ranks)
  {
    // nparts is the number of groups the input communicator is split into, not the
    // number of ranks in one, so 1 gives a single group spanning every rank. That is
    // what this constructor wants: its communicator covers
    // distributed_ranks * shared_ranks ranks, and distributed_ranks is derived from
    // it below. Passing the world size instead produced that many groups of one rank
    // each, which left comm.size() at 1 and made every shared_ranks > 1 call return
    // at the guard without testing anything.
    auto comm_distributed = std::make_unique<Communicate>(*OHMMS::Controller, 1);

    auto& comm(*comm_distributed);
    REQUIRE(comm.size() == OHMMS::Controller->size());

    // need sufficient number of ranks to test the distributing and/or sharing feature.
    if (comm.size() % shared_ranks > 0)
      return;

    MultiBsplineMPIShared<T> bs(grid, bc, num_splines, std::move(comm_distributed), comm.size() / shared_ranks);

    const size_t npad = getAlignedSize<T>(num_splines);
    REQUIRE(bs.num_splines_padded() == getAlignedSize<T>(num_splines));

    UBspline_3d_d* aspline = create_UBspline_3d_d(grid[0], grid[1], grid[2], bc[0], bc[1], bc[2], data.data());

    auto offsets = FairDivideAligned<std::vector<size_t>>(num_splines, getAlignment<T>(), comm.size());
    for (int i = offsets[comm.rank()]; i < offsets[comm.rank() + 1]; i++)
      bs.set_spline(*aspline, i);
    comm.barrier();

    destroy_Bspline(aspline);

    //  Code from here to the end of the function is generated by gen_bspline_values.py

    TinyVector<T, 3> pos = {0, 0, 0};

    // symbolic value at pos =  (cx[0]/6 + 2*cx[1]/3 + cx[2]/6)*(cy[0]/6 + 2*cy[1]/3 + cy[2]/6)*(cz[0]/6 + 2*cz[1]/3 + cz[2]/6)
    aligned_vector<T> v(npad);
    bs.evaluate_v(pos, v);
    CHECK(v[0] == Approx(-3.529930688e-12));

    VectorSoaContainer<T, 3> dv(npad);
    VectorSoaContainer<T, 6> hess(npad);
    bs.evaluate_vgh(pos, v, dv, hess);
    // Gradient
    CHECK(dv[0][0] == Approx(6.178320809));
    CHECK(dv[0][1] == Approx(-7.402942564));
    CHECK(dv[0][2] == Approx(-6.178320809));

    // Hessian
    for (int i = 0; i < 6; i++)
    {
      CHECK(hess[0][i] == Approx(0.0));
    }

    pos = {0.1, 0.2, 0.3};
    bs.evaluate_v(pos, v);

    // Value
    CHECK(v[0] == Approx(-0.9476393279));

    bs.evaluate_vgh(pos, v, dv, hess);
    // Value
    CHECK(v[0] == Approx(-0.9476393279));
    // Gradient
    CHECK(dv[0][0] == Approx(5.111042137));
    CHECK(dv[0][1] == Approx(5.989106342));
    CHECK(dv[0][2] == Approx(1.952244379));
    // Hessian
    CHECK(hess[0][0] == Approx(-21.34557341));
    CHECK(hess[0][1] == Approx(1.174505743e-09));
    CHECK(hess[0][2] == Approx(-1.1483271e-09));
    CHECK(hess[0][3] == Approx(133.9204891));
    CHECK(hess[0][4] == Approx(-2.15319293e-09));
    CHECK(hess[0][5] == Approx(34.53786329));


    VectorSoaContainer<T, 3> lap(npad);
    bs.evaluate_vgl(pos, v, dv, lap);
    // Value
    CHECK(v[0] == Approx(-0.9476393279));
    // Gradient
    CHECK(dv[0][0] == Approx(5.111042137));
    CHECK(dv[0][1] == Approx(5.989106342));
    CHECK(dv[0][2] == Approx(1.952244379));
    // Laplacian
    CHECK(lap[0][0] == Approx(147.1127789));

    VectorSoaContainer<T, 10> ghess(npad);
    bs.evaluate_vghgh(pos, v, dv, hess, ghess);
    // Value
    CHECK(v[0] == Approx(-0.9476393279));
    // Gradient
    CHECK(dv[0][0] == Approx(5.111042137));
    CHECK(dv[0][1] == Approx(5.989106342));
    CHECK(dv[0][2] == Approx(1.952244379));
    // Hessian
    CHECK(hess[0][0] == Approx(-21.34557341));
    CHECK(hess[0][1] == Approx(1.174505743e-09));
    CHECK(hess[0][2] == Approx(-1.1483271e-09));
    CHECK(hess[0][3] == Approx(133.9204891));

    CHECK(hess[0][4] == Approx(-2.15319293e-09));
    CHECK(hess[0][5] == Approx(34.53786329));


    // Catch default is 100*(float epsilson)
    double eps = 2000 * std::numeric_limits<float>::epsilon();

    // Gradient of Hessian
    CHECK(ghess[0][0] == Approx(-213.455734));
    CHECK(ghess[0][1] == Approx(2.311193459e-09).epsilon(eps));
    CHECK(ghess[0][2] == Approx(3.468205279e-09).epsilon(eps));
    CHECK(ghess[0][3] == Approx(1.58092329e-07).epsilon(eps));
    CHECK(ghess[0][4] == Approx(1.255694171e-08).epsilon(eps));
    CHECK(ghess[0][5] == Approx(4.78981157e-08).epsilon(eps));
    CHECK(ghess[0][6] == Approx(-1753.041961));
    CHECK(ghess[0][7] == Approx(-2.575826885e-09).epsilon(eps));
    CHECK(ghess[0][8] == Approx(-4.683496702e-09).epsilon(eps));
    CHECK(ghess[0][9] == Approx(-81.53283531));

    MultiBsplineOffloadMapper<T> mapped_bs(bs);
    mapped_bs.mapToDevice();
    mapped_bs.updateToDevice();

    const int num_pos = 3;
    Vector<T, OffloadAllocator<T>> pos_arr{0.1, 0.2, 0.3, 0.3, 0.1, 0.2, 0.1, 0.2, 0.3};
    pos_arr.updateTo();

    auto num_splines_padded = bs.num_splines_padded();

    Vector<T, OffloadAllocator<T>> spline_v_vals(num_pos * num_splines_padded);
    mapped_bs.mw_evaluate_v(num_pos, pos_arr.data(), spline_v_vals.data(), num_splines_padded);
    spline_v_vals.updateFrom();

    CHECK(spline_v_vals[0] == Approx(-0.9476393279));
    CHECK(spline_v_vals[num_splines_padded * 2] == Approx(-0.9476393279));

    Vector<T, OffloadAllocator<T>> spline_vgh_vals(num_pos * num_splines_padded * SoAFields3D::NUM_FIELDS);
    mapped_bs.mw_evaluate_vgh(num_pos, pos_arr.data(), spline_vgh_vals.data(),
                              num_splines_padded * SoAFields3D::NUM_FIELDS, num_splines_padded);
    spline_vgh_vals.updateFrom();

    CHECK(spline_vgh_vals[0] == Approx(-0.9476393279));
    CHECK(spline_vgh_vals[num_splines_padded * SoAFields3D::GRAD1] == Approx(5.989106342));
    CHECK(spline_vgh_vals[num_splines_padded * SoAFields3D::HESS22] == Approx(34.53786329));

    Vector<T, OffloadAllocator<T>>
        spline_vgh_vals_w2(spline_vgh_vals, &spline_vgh_vals[num_splines_padded * SoAFields3D::NUM_FIELDS * 2],
                           num_splines_padded * SoAFields3D::NUM_FIELDS);
    CHECK(spline_vgh_vals_w2[0] == Approx(-0.9476393279));
    CHECK(spline_vgh_vals_w2[num_splines_padded * SoAFields3D::GRAD1] == Approx(5.989106342));
    CHECK(spline_vgh_vals_w2[num_splines_padded * SoAFields3D::HESS22] == Approx(34.53786329));
  }
};

/** Coefficients in a shared window, evaluated through a device mapping.
 *
 * Covers the path MultiBsplineMPISharedOffload puts into production: the coefficients
 * are allocated once per group of ranks in an MPI-3 shared window, mapped onto the
 * device through its owned mapper, and read back by multi-walker evaluation. The values are
 * compared against the host evaluation of the same object, so the check is on the
 * mapping rather than on any hard-coded number.
 *
 * Host builds exercise the shared window and blocked evaluation with inactive offload
 * pragmas.
 */
template<typename T>
struct test_shared_offload : public test_splines_base<T, 5>
{
  using base = test_splines_base<T, 5>;
  using base::bc;
  using base::data;
  using base::grid;

  void test(size_t num_splines, unsigned shared_ranks, unsigned distributed_ranks = 1)
  {
    // the second argument of Communicate(const Communicate&, int nparts, int) is the
    // number of groups, not the size of one, so a group of this size needs
    // world/(distributed*shared) parts. Passing the world size instead yields that many
    // groups of one rank each, comm.size() == 1 everywhere, and nothing to test.
    const int world      = OHMMS::Controller->size();
    const unsigned group = distributed_ranks * shared_ranks;
    if (world % group > 0 || num_splines < distributed_ranks)
      return;
    auto comm_shared = std::make_unique<Communicate>(*OHMMS::Controller, world / group);
    auto& comm(*comm_shared);
    REQUIRE(comm.size() == static_cast<int>(group));

    MultiBsplineMPISharedOffload<T> bs(grid, bc, num_splines, std::move(comm_shared), distributed_ranks);
    REQUIRE(bs.getNumBlocks() == distributed_ranks);

    const size_t npad = getAlignedSize<T>(num_splines);
    REQUIRE(bs.num_splines_padded() == npad);

    UBspline_3d_d* aspline = create_UBspline_3d_d(grid[0], grid[1], grid[2], bc[0], bc[1], bc[2], data.data());
    auto offsets           = FairDivideAligned<std::vector<size_t>>(num_splines, getAlignment<T>(), comm.size());
    for (int i = offsets[comm.rank()]; i < offsets[comm.rank() + 1]; i++)
      bs.set_spline(*aspline, i);
    destroy_Bspline(aspline);

    // publishes every rank's stores before uploading the shared coefficients and
    // repairing the device coefs pointer
    bs.finalize();

    const TinyVector<T, 3> pos = {0.1, 0.2, 0.3};
    aligned_vector<T> v_host(npad);
    bs.evaluate_v(pos, v_host);

    Vector<T, OffloadAllocator<T>> pos_arr{pos[0], pos[1], pos[2]};
    pos_arr.updateTo();
    Vector<T, OffloadAllocator<T>> v_dev(npad);
    bs.mw_evaluate_v(1, pos_arr.data(), v_dev.data(), npad);
    v_dev.updateFrom();

    for (size_t i = 0; i < num_splines; i++)
      CHECK(v_dev[i] == Approx(v_host[i]));
  }
};

/** Coefficients split into blocks across ranks, evaluated on the device.
 *
 * Distributing splits orbitals into blocks. MultiBsplineOffloadMapper::mw_evaluate_v
 * walks the blocks, takes each block's descriptor and coefficients, and writes results
 * at the corresponding global offset.
 *
 * The device result is compared against the host evaluation of the same object, which
 * walks the blocks too, so what is under test is that the blocked device path agrees
 * with the blocked host path rather than any hard-coded value.
 */
template<typename T>
struct test_distributed_offload : public test_splines_base<T, 5>
{
  using base = test_splines_base<T, 5>;
  using base::bc;
  using base::data;
  using base::grid;

  void test(size_t num_splines, unsigned distributed_ranks)
  {
    const int world = OHMMS::Controller->size();
    if (world % distributed_ranks > 0 || num_splines < distributed_ranks)
      return;

    // one group spanning every rank, as MultiBsplineMPIShared expects
    auto comm_distributed = std::make_unique<Communicate>(*OHMMS::Controller, 1);
    auto& comm(*comm_distributed);

    MultiBsplineMPIShared<T> bs(grid, bc, num_splines, std::move(comm_distributed), distributed_ranks);
    REQUIRE(bs.getNumBlocks() == distributed_ranks);

    const size_t npad      = getAlignedSize<T>(num_splines);
    UBspline_3d_d* aspline = create_UBspline_3d_d(grid[0], grid[1], grid[2], bc[0], bc[1], bc[2], data.data());
    auto offsets           = FairDivideAligned<std::vector<size_t>>(num_splines, getAlignment<T>(), comm.size());
    for (int i = offsets[comm.rank()]; i < offsets[comm.rank() + 1]; i++)
      bs.set_spline(*aspline, i);
    comm.barrier();
    destroy_Bspline(aspline);

    const TinyVector<T, 3> pos = {0.1, 0.2, 0.3};
    aligned_vector<T> v_host(npad);
    bs.evaluate_v(pos, v_host);

    MultiBsplineOffloadMapper<T> mapped_bs(bs);
    mapped_bs.mapToDevice();
    mapped_bs.updateToDevice();

    Vector<T, OffloadAllocator<T>> pos_arr{pos[0], pos[1], pos[2]};
    pos_arr.updateTo();
    Vector<T, OffloadAllocator<T>> v_dev(npad);
    mapped_bs.mw_evaluate_v(1, pos_arr.data(), v_dev.data(), npad);
    v_dev.updateFrom();

    for (size_t i = 0; i < num_splines; i++)
      CHECK(v_dev[i] == Approx(v_host[i]));
  }
};

/** One device copy of the coefficients, read by every rank in the group.
 *
 * MultiBsplineOffloadMapperPeer has the owning rank allocate the device memory and
 * export a handle, the others open it, and every rank binds it to its own host pointer
 * with omp_target_associate_ptr. The evaluation is then the ordinary one: what is being
 * checked is that a rank reading through a pointer it did not allocate gets the same
 * values as the host evaluation.
 *
 * Configurations without interoperable device handles use the per-rank mapping.
 */
template<typename T>
struct test_peer_offload : public test_splines_base<T, 5>
{
  using base = test_splines_base<T, 5>;
  using base::bc;
  using base::data;
  using base::grid;

  void test(size_t num_splines, unsigned distributed_ranks = 1)
  {
    const int world = OHMMS::Controller->size();
    if (world % distributed_ranks > 0 || num_splines < distributed_ranks)
      return;
    auto comm_shared = std::make_unique<Communicate>(*OHMMS::Controller, 1);
    auto& comm(*comm_shared);

    MultiBsplineMPIShared<T> bs(grid, bc, num_splines, std::move(comm_shared), distributed_ranks);
    REQUIRE(bs.getNumBlocks() == distributed_ranks);

    if (distributed_ranks > 1)
    {
      const size_t alignment = getAlignment<T>();
      REQUIRE(num_splines == (distributed_ranks - 1) * alignment + 2);

      const auto& block_starts = bs.getBlockOffsets();
      REQUIRE(block_starts.size() == distributed_ranks + 1);
      for (size_t ib = 0; ib + 1 < distributed_ranks; ++ib)
      {
        REQUIRE(block_starts[ib] == ib * alignment);
        REQUIRE(bs.getBlock(ib).num_splines == alignment);
        REQUIRE(bs.getBlock(ib).z_stride == alignment);
      }
      REQUIRE(block_starts[distributed_ranks - 1] == (distributed_ranks - 1) * alignment);
      REQUIRE(block_starts[distributed_ranks] == num_splines);
      REQUIRE(bs.getBlock(distributed_ranks - 1).num_splines == 2);
      REQUIRE(bs.getBlock(distributed_ranks - 1).z_stride == alignment);
      REQUIRE(bs.num_splines_padded() == distributed_ranks * alignment);
    }

    const size_t npad      = getAlignedSize<T>(num_splines);
    UBspline_3d_d* aspline = create_UBspline_3d_d(grid[0], grid[1], grid[2], bc[0], bc[1], bc[2], data.data());
    auto offsets           = FairDivideAligned<std::vector<size_t>>(num_splines, getAlignment<T>(), comm.size());
    for (int i = offsets[comm.rank()]; i < offsets[comm.rank() + 1]; i++)
      bs.set_spline(*aspline, i);
    comm.barrier();
    destroy_Bspline(aspline);

    const TinyVector<T, 3> pos = {0.1, 0.2, 0.3};
    aligned_vector<T> v_host(npad);
    bs.evaluate_v(pos, v_host);

    MultiBsplineOffloadMapperPeer<T> mapped_bs(bs, comm);
    mapped_bs.mapToDevice();
    mapped_bs.updateToDevice();

    // Production kernels reach coefficients through the mapped spline descriptor.
    // Reading through that pointer checks that the mapper attaches the IPC allocation
    // to the descriptor rather than only making the standalone host pointer present.
    const auto* spline_ptr = &bs.getBlock(0);
    T device_coef{};
    PRAGMA_OFFLOAD("omp target map(from: device_coef)") { device_coef = spline_ptr->coefs[0]; }
    CHECK(device_coef == Approx(spline_ptr->coefs[0]));

    Vector<T, OffloadAllocator<T>> pos_arr{pos[0], pos[1], pos[2]};
    pos_arr.updateTo();
    Vector<T, OffloadAllocator<T>> v_dev(npad);
    mapped_bs.mw_evaluate_v(1, pos_arr.data(), v_dev.data(), npad);
    v_dev.updateFrom();

    for (size_t i = 0; i < num_splines; i++)
      CHECK(v_dev[i] == Approx(v_host[i]));
  }
};

TEST_CASE("MultiBsplineOffloadMapperPeer shared device copy", "[spline2][shared-offload]")
{
  test_peer_offload<double>().test(13);
  test_peer_offload<float>().test(11);
  // more than one block, so ownership rotates and every rank allocates a share
  test_peer_offload<double>().test(getAlignment<double>() + 2, 2);
  test_peer_offload<float>().test(getAlignment<float>() + 2, 2);
  test_peer_offload<double>().test(3 * getAlignment<double>() + 2, 4);
}

TEST_CASE("MultiBsplineMPIShared distributed offload double", "[spline2]")
{
  test_distributed_offload<double>().test(13, 1);
  test_distributed_offload<double>().test(13, 2);
  test_distributed_offload<double>().test(13, 4);
}

TEST_CASE("MultiBsplineMPIShared distributed offload float", "[spline2]")
{
  test_distributed_offload<float>().test(11, 1);
  test_distributed_offload<float>().test(11, 2);
  test_distributed_offload<float>().test(11, 4);
}

TEST_CASE("MultiBsplineMPISharedOffload periodic double", "[spline2][shared-offload]")
{
  test_shared_offload<double>().test(13, 1);
  test_shared_offload<double>().test(13, 2);
  // shared and distributed together: two blocks, each shared across two ranks
  test_shared_offload<double>().test(13, 2, 2);
  test_shared_offload<double>().test(13, 1, 2);
}

TEST_CASE("MultiBsplineMPISharedOffload periodic float", "[spline2][shared-offload]")
{
  test_shared_offload<float>().test(11, 1);
  test_shared_offload<float>().test(11, 2);
  test_shared_offload<float>().test(11, 2, 2);
  test_shared_offload<float>().test(11, 1, 2);
}

TEST_CASE("MultiBsplineMPIShared periodic double", "[spline2]")
{
  test_splines<double>().test(13, 1);
  test_splines<double>().test(13, 2);
  test_splines<double>().test(13, 3);
}

TEST_CASE("MultiBsplineMPIShared periodic float", "[spline2]")
{
  test_splines<float>().test(11, 1);
  test_splines<float>().test(11, 2);
  test_splines<float>().test(11, 3);
}

} // namespace qmcplusplus
