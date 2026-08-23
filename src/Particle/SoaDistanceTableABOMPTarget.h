//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//                    Amrita Mathuriya, amrita.mathuriya@intel.com, Intel Corp.
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////
// -*- C++ -*-
#ifndef QMCPLUSPLUS_DTDIMPL_AB_OMPTARGET_H
#define QMCPLUSPLUS_DTDIMPL_AB_OMPTARGET_H

#include <algorithm>
#include "Lattice/ParticleBConds3DSoa.h"
#include "DistanceTable.h"
#include "OMPTarget/OffloadAlignedAllocators.hpp"
#include "Particle/RealSpacePositionsOMPTarget.h"
#include "ResourceCollection.h"
#include "OMPTarget/OMPTargetMath.hpp"

namespace qmcplusplus
{
/**@ingroup nnlist
 * @brief A derived classe from DistacneTableData, specialized for AB using a transposed form
 */
template<typename T, unsigned D, int SC>
class SoaDistanceTableABOMPTarget : public DTD_BConds<T, D, SC>, public DistanceTableAB
{
private:
  template<typename DT>
  using OffloadPinnedVector = Vector<DT, OffloadPinnedAllocator<DT>>;

  ///accelerator output buffer for r and dr
  OffloadPinnedVector<RealType> r_dr_memorypool_;
  ///accelerator input array for a list of target particle positions, num_targets_ x D
  OffloadPinnedVector<T> target_pos;

  ///multi walker shared memory buffer
  struct DTABMultiWalkerMem : public Resource
  {
    ///accelerator output array for multiple walkers, [1+D][num_targets_][num_padded] (distances, displacements)
    OffloadPinnedVector<T> mw_r_dr;
    ///accelerator input buffer for multiple data set
    OffloadPinnedVector<char> offload_input;
    /** temporary distances and displacements of one moved target against every source,
     *  for a whole batch: [new: nw][old: nw], each [1+D][num_sources padded]
     *
     *  A one-body Jastrow reduces over the sources of a single target, which is the same
     *  shape the two-body one reduces over electrons, and that one is handed the batch at
     *  once because the electron-electron table keeps this on the device. This is the
     *  matching storage for the electron-ion table.
     */
    OffloadPinnedVector<T> mw_new_old_dist_displ;
    ///device pointers of the moved target's position per walker
    OffloadPinnedVector<char> move_input;

    DTABMultiWalkerMem() : Resource("DTABMultiWalkerMem") {}

    DTABMultiWalkerMem(const DTABMultiWalkerMem&) : DTABMultiWalkerMem() {}

    std::unique_ptr<Resource> makeClone() const override { return std::make_unique<DTABMultiWalkerMem>(*this); }
  };

  ResourceHandle<DTABMultiWalkerMem> mw_mem_handle_;

  static void associateResource(const RefVectorWithLeader<DistanceTable>& dt_list,
                                const RefVectorWithLeader<const DynamicCoordinates>& coords_list)
  {
    auto& dt_leader = dt_list.getCastedLeader<SoaDistanceTableABOMPTarget>();

    // initialize memory containers and views
    size_t count_targets = 0;
    for (size_t iw = 0; iw < dt_list.size(); iw++)
    {
      auto& dt                         = dt_list.getCastedElement<SoaDistanceTableABOMPTarget>(iw);
      count_targets += dt.num_targets_ = coords_list[iw].size();
      dt.r_dr_memorypool_.free();
    }

    const size_t num_sources   = dt_leader.num_sources_;
    const size_t num_padded    = getAlignedSize<T>(dt_leader.num_sources_);
    const size_t stride_size   = num_padded * (D + 1);
    const size_t total_targets = count_targets;
    auto& mw_r_dr              = dt_leader.mw_mem_handle_.getResource().mw_r_dr;
    mw_r_dr.resize(total_targets * stride_size);

    count_targets = 0;
    for (size_t iw = 0; iw < dt_list.size(); iw++)
    {
      auto& dt = dt_list.getCastedElement<SoaDistanceTableABOMPTarget>(iw);
      assert(num_sources == dt.num_sources_);

      /* Grow by clearing first. These elements are views attached into mw_r_dr, and a
       * std::vector growing past its capacity copies what it holds; Vector's copy
       * constructor allocates, so the survivors would come back owning memory and
       * attachReference below rejects them. Target counts are constant in the usual
       * paths, so this only bites when a walker's count changes between calls.
       */
      if (dt.distances_.size() != static_cast<size_t>(dt.targets()))
      {
        dt.distances_.clear();
        dt.displacements_.clear();
        dt.distances_.resize(dt.targets());
        dt.displacements_.resize(dt.targets());
      }

      for (int i = 0; i < dt.targets(); ++i)
      {
        dt.distances_[i].attachReference(mw_r_dr.data() + (i + count_targets) * stride_size, num_sources);
        dt.displacements_[i].attachReference(num_sources, num_padded,
                                             mw_r_dr.data() + (i + count_targets) * stride_size + num_padded);
      }
      count_targets += dt.targets();
    }
  }

public:
  SoaDistanceTableABOMPTarget(const ParticleSet& source, const std::string& target_name)
      : DTD_BConds<T, D, SC>(source.getLattice()),
        DistanceTableAB(source, target_name, DTModes::ALL_OFF),
        offload_timer_(createGlobalTimer("DTABOMPTarget::offload_" + name_, timer_level_fine)),
        evaluate_timer_(createGlobalTimer("DTABOMPTarget::evaluate_" + name_, timer_level_fine))

  {
    auto* coordinates_soa = dynamic_cast<const RealSpacePositionsOMPTarget*>(&source.getCoordinates());
    if (!coordinates_soa)
      throw std::runtime_error("Source particle set doesn't have OpenMP offload. Contact developers!");
    PRAGMA_OFFLOAD("omp target enter data map(to : this[:1])")

    // The padding of temp_r_ and temp_dr_ is necessary for the memory copy in the update function
    // temp_r_ is padded explicitly while temp_dr_ is padded internally
    const int num_padded = getAlignedSize<T>(num_sources_);
    temp_r_.resize(num_padded);
    temp_dr_.resize(num_sources_);
  }

  SoaDistanceTableABOMPTarget()                                   = delete;
  SoaDistanceTableABOMPTarget(const SoaDistanceTableABOMPTarget&) = delete;

  ~SoaDistanceTableABOMPTarget() { PRAGMA_OFFLOAD("omp target exit data map(delete : this[:1])") }

  void createResource(ResourceCollection& collection) const override
  {
    auto resource_index = collection.addResource(std::make_unique<DTABMultiWalkerMem>());
  }

  void acquireResource(ResourceCollection& collection, const RefVectorWithLeader<DistanceTable>& dt_list) const override
  {
    auto& dt_leader          = dt_list.getCastedLeader<SoaDistanceTableABOMPTarget>();
    dt_leader.mw_mem_handle_ = collection.lendResource<DTABMultiWalkerMem>();
  }

  void releaseResource(ResourceCollection& collection, const RefVectorWithLeader<DistanceTable>& dt_list) const override
  {
    /* num_targets_ is what targets() reports, so it is not scratch to invalidate here.
     * associateResource assigns it from the coordinates on every multi-walker
     * evaluation, which is where the count is established. Zeroing it on acquire and
     * release only made the table report no targets to anything asking between a
     * release and the next batched section, which a single-walker consumer does.
     */
    auto& dt_leader = dt_list.getCastedLeader<SoaDistanceTableABOMPTarget>();
    collection.takebackResource(dt_leader.mw_mem_handle_);
  }

  const T* getMultiWalkerDataPtr() const override { return mw_mem_handle_.getResource().mw_r_dr.data(); }

  const RealType* getMultiWalkerTempDataPtr() const override
  { return mw_mem_handle_.getResource().mw_new_old_dist_displ.data(); }

  /** the device address of the same buffer
   *
   * The electron electron table attaches each walker's temporary arrays into its multi
   * walker buffer, so the host side of that buffer is filled by the per walker move and a
   * consumer naming the host address reads valid data. This table keeps them separate:
   * only the device side is written, so a consumer has to name the device address.
   */
  const RealType* getMultiWalkerTempDeviceDataPtr() const override
  { return mw_mem_handle_.getResource().mw_new_old_dist_displ.device_data(); }

  void requireTempDataOnDevice() const override { temp_data_on_device_ = true; }

  /** whether the device side holds the temp distances for the move in hand
   *
   * Asking for them does not produce them: the batched move fills the device side only when
   * the request was already standing when it ran, and the move for the current particle has
   * happened by the time a consumer asks. Reporting the request here would tell that consumer
   * the buffer is good on the one move between the request and the first fill, where it holds
   * whatever it held before, and a zero distance there divides.
   */
  bool hasTempDataOnDevice() const override { return temp_data_filled_on_device_; }

  size_t getPerTargetPctlStrideSize() const override { return getAlignedSize<T>(num_sources_) * (D + 1); }

  /** evaluate the full table */
  inline void evaluate(const DynamicCoordinates& coords) override
  {
    if (num_targets_ != coords.size())
      resize(coords.size());

    ScopedTimer local_timer(evaluate_timer_);
    // be aware of the sign of Displacement
    const int num_targets_local = num_targets_;
    const int num_sources_local = num_sources_;
    const int num_padded        = getAlignedSize<T>(num_sources_);

    auto& positions = coords.getAllParticlePos();
    target_pos.resize(num_targets_ * D);
    for (size_t iat = 0; iat < num_targets_; iat++)
      for (size_t idim = 0; idim < D; idim++)
        target_pos[iat * D + idim] = positions[iat][idim];

    auto* target_pos_ptr = target_pos.data();
    auto* source_pos_ptr = origin_.getCoordinates().getAllParticlePos().data();
    auto* r_dr_ptr       = distances_[0].data();
    assert(distances_[0].data() + num_padded == displacements_[0].data());

    // To maximize thread usage, the loop over electrons is chunked. Each chunk is sent to an OpenMP offload thread team.
    const int ChunkSizePerTeam = 512;
    const size_t num_teams     = (num_sources_ + ChunkSizePerTeam - 1) / ChunkSizePerTeam;
    const size_t stride_size   = getPerTargetPctlStrideSize();

    {
      ScopedTimer offload(offload_timer_);
      PRAGMA_OFFLOAD("omp target teams distribute collapse(2) num_teams(num_targets_*num_teams) \
                        map(to: source_pos_ptr[:num_padded*D]) \
                        map(always, to: target_pos_ptr[:num_targets_*D]) \
                        map(always, from: r_dr_ptr[:num_targets_*stride_size])")
      for (int iat = 0; iat < num_targets_local; ++iat)
        for (int team_id = 0; team_id < num_teams; team_id++)
        {
          const int first = ChunkSizePerTeam * team_id;
          const int last  = omptarget::min(first + ChunkSizePerTeam, num_sources_local);

          T pos[D];
          for (int idim = 0; idim < D; idim++)
            pos[idim] = target_pos_ptr[iat * D + idim];

          auto* r_iat_ptr  = r_dr_ptr + iat * stride_size;
          auto* dr_iat_ptr = r_iat_ptr + num_padded;

          PRAGMA_OFFLOAD("omp parallel for")
          for (int iel = first; iel < last; iel++)
            DTD_BConds<T, D, SC>::computeDistancesOffload(pos, source_pos_ptr, num_padded, r_iat_ptr, dr_iat_ptr,
                                                          num_padded, iel);
        }
    }
  }

  inline void mw_evaluate(const RefVectorWithLeader<DistanceTable>& dt_list,
                          const RefVectorWithLeader<const DynamicCoordinates>& coords_list) const override
  {
    assert(this == &dt_list.getLeader());
    auto& dt_leader = dt_list.getCastedLeader<SoaDistanceTableABOMPTarget>();

    ScopedTimer local_timer(evaluate_timer_);

    associateResource(dt_list, coords_list);
    const size_t nw            = dt_list.size();
    DTABMultiWalkerMem& mw_mem = dt_leader.mw_mem_handle_;
    auto& mw_r_dr              = mw_mem.mw_r_dr;

    size_t count_targets = 0;
    for (const DynamicCoordinates& coords : coords_list)
      count_targets += coords.size();
    const size_t total_targets = count_targets;

    const int num_padded = getAlignedSize<T>(num_sources_);

#ifndef NDEBUG
    const int stride_size = getPerTargetPctlStrideSize();
    count_targets         = 0;
    for (size_t iw = 0; iw < dt_list.size(); iw++)
    {
      auto& dt = dt_list.getCastedElement<SoaDistanceTableABOMPTarget>(iw);

      for (int i = 0; i < dt.targets(); ++i)
      {
        assert(dt.distances_[i].data() == mw_r_dr.data() + (i + count_targets) * stride_size);
        assert(dt.displacements_[i].data() == mw_r_dr.data() + (i + count_targets) * stride_size + num_padded);
      }
      count_targets += dt.targets();
    }
#endif

    // This is horrible optimization putting different data types in a single buffer but allows a single H2D transfer
    const size_t realtype_size = sizeof(RealType);
    const size_t int_size      = sizeof(int);
    const size_t ptr_size      = sizeof(RealType*);
    auto& offload_input        = mw_mem.offload_input;
    offload_input.resize(total_targets * D * realtype_size + total_targets * int_size + nw * ptr_size);
    auto source_ptrs      = reinterpret_cast<RealType**>(offload_input.data());
    auto target_positions = reinterpret_cast<RealType*>(offload_input.data() + ptr_size * nw);
    auto walker_id_ptr =
        reinterpret_cast<int*>(offload_input.data() + ptr_size * nw + total_targets * D * realtype_size);

    count_targets = 0;
    for (size_t iw = 0; iw < nw; iw++)
    {
      auto& dt = dt_list.getCastedElement<SoaDistanceTableABOMPTarget>(iw);
      auto& coords(coords_list[iw]);

      assert(dt.targets() == coords.size());
      assert(num_sources_ == dt.num_sources_);

      auto& RSoA_OMPTarget = dynamic_cast<const RealSpacePositionsOMPTarget&>(dt.origin_.getCoordinates());
      source_ptrs[iw]      = const_cast<RealType*>(RSoA_OMPTarget.getDevicePtr());

      auto& positions = coords.getAllParticlePos();
      for (size_t iat = 0; iat < coords.size(); ++iat, ++count_targets)
      {
        walker_id_ptr[count_targets] = iw;
        for (size_t idim = 0; idim < D; idim++)
          target_positions[count_targets * D + idim] = positions[iat][idim];
      }
    }

    /* One team per target, looping its threads over the sources, leaves almost every
     * thread idle whenever a cell has few ions: a team is hundreds of threads wide and a
     * virtual particle set against a two atom cell gives it two distances to compute. The
     * target and source axes are independent, one output element each, so flattening them
     * into a single iteration space fills the teams from the product instead of from the
     * source count alone. Consecutive work items keep the same target and walk the
     * sources, which is the order the distance and displacement rows are written in.
     * Stating the collapse rather than computing a flat index keeps a division and a
     * modulo out of every iteration, which matters for cells with more sources than a
     * team is wide, the ones that already filled their teams under the chunking.
     */
    auto* r_dr_ptr              = mw_r_dr.data();
    auto* input_ptr             = offload_input.data();
    const int num_sources_local = num_sources_;

    /* Collapsing fills the teams but leaves the count to the runtime, which packs a
     * moderate collapsed space into too few of them: measured against the chunking it runs
     * at 0.92 and 0.97 of its speed at 512 targets by 128 and by 512 sources while winning
     * the other ten cells of the sweep. Naming a count derived from the work keeps the grid
     * wide there too.
     */
    const long total_work = static_cast<long>(total_targets) * num_sources_local;
    const int num_teams   = static_cast<int>(std::min<long>(std::max<long>(total_work / 64, 1), 65535));

    {
      ScopedTimer offload(dt_leader.offload_timer_);
      PRAGMA_OFFLOAD("omp target teams distribute parallel for collapse(2) num_teams(num_teams) \
                          map(always, to: input_ptr[:offload_input.size()]) \
                          depend(out:r_dr_ptr[:mw_r_dr.size()])")
      for (int iat = 0; iat < total_targets; ++iat)
        for (int iel = 0; iel < num_sources_local; ++iel)
        {
          auto* target_pos_ptr = reinterpret_cast<RealType*>(input_ptr + ptr_size * nw);
          const int walker_id =
              reinterpret_cast<int*>(input_ptr + ptr_size * nw + total_targets * D * realtype_size)[iat];
          auto* source_pos_ptr = reinterpret_cast<RealType**>(input_ptr)[walker_id];
          auto* r_iat_ptr      = r_dr_ptr + iat * num_padded * (D + 1);
          auto* dr_iat_ptr     = r_dr_ptr + iat * num_padded * (D + 1) + num_padded;

          T pos[D];
          for (int idim = 0; idim < D; idim++)
            pos[idim] = target_pos_ptr[iat * D + idim];

          DTD_BConds<T, D, SC>::computeDistancesOffload(pos, source_pos_ptr, num_padded, r_iat_ptr, dr_iat_ptr,
                                                        num_padded, iel);
        }

      if (!(modes_ & DTModes::MW_EVALUATE_RESULT_NO_TRANSFER_TO_HOST))
      {
        PRAGMA_OFFLOAD(
            "omp target update from(r_dr_ptr[:mw_r_dr.size()]) depend(inout:r_dr_ptr[:mw_r_dr.size()]) nowait")
      }
      // wait for computing and (optional) transferring back to host.
      // It can potentially be moved to ParticleSet to fuse multiple similar taskwait
      PRAGMA_OFFLOAD("omp taskwait")
    }
  }

  inline void mw_recompute(const RefVectorWithLeader<DistanceTable>& dt_list,
                           const RefVectorWithLeader<ParticleSet>& p_list,
                           const std::vector<bool>& recompute) const override
  {
    DistanceTable::mw_evaluate(dt_list, p_list);
  }

  ///evaluate the temporary pair relations
  inline void move(const ParticleSet& P, const PosType& rnew, const IndexType iat, bool prepare_old) override
  {
    // this form does not touch the device side, so what is there no longer describes the move
    temp_data_filled_on_device_ = false;
    // Single particle against all sources is cheap and is computed on the host, as
    // SoaDistanceTableAAOMPTarget::move does. The device side is not written here: the
    // full table is recomputed by mw_evaluate, so anything stored into distances_ or
    // displacements_ (which alias the device-mapped mw_r_dr) is for host consumers
    // only and must not be relied on inside a target region.
    DTD_BConds<T, D, SC>::computeDistances(rnew, origin_.getCoordinates().getAllParticlePos(), temp_r_.data(),
                                           temp_dr_, 0, num_sources_);
    if (!(modes_ & DTModes::NEED_FULL_TABLE_ANYTIME) && prepare_old)
      DTD_BConds<T, D, SC>::computeDistances(P.R[iat], origin_.getCoordinates().getAllParticlePos(),
                                             distances_[iat].data(), displacements_[iat], 0, num_sources_);
  }

  /** the moved target of every walker against every source, computed on the device
   *
   * The base form loops the single walker move, which leaves the result on the host only,
   * so a component reducing over sources for the batch has nothing on the device to read
   * and falls back to a walker at a time. The electron-electron table already keeps this;
   * this is the same for the electron-ion one.
   *
   * Host data follows only when DTModes::NEED_TEMP_DATA_ON_HOST is set, as in the
   * electron-electron table, so a consumer reading getTempDists() keeps working.
   */
  void mw_move(const RefVectorWithLeader<DistanceTable>& dt_list,
               const RefVectorWithLeader<ParticleSet>& p_list,
               const std::vector<PosType>& rnew_list,
               const IndexType iat,
               bool prepare_old = true) const override
  {
    assert(this == &dt_list.getLeader());
    auto& dt_leader             = dt_list.getCastedLeader<SoaDistanceTableABOMPTarget>();
    auto& mw_mem                = dt_leader.mw_mem_handle_.getResource();
    auto& mw_new_old_dist_displ = mw_mem.mw_new_old_dist_displ;
    auto& move_input            = mw_mem.move_input;
    const size_t nw             = dt_list.size();

    const size_t num_padded  = getAlignedSize<T>(num_sources_);
    const size_t stride_size = num_padded * (D + 1);
    mw_new_old_dist_displ.resize(2 * nw * stride_size);

    /* One pointer to the walker's sources and one new position per walker, packed so the
     * kernel takes a single mapped buffer rather than one clause per array.
     */
    const size_t ptr_size      = sizeof(RealType*);
    const size_t realtype_size = sizeof(RealType);
    move_input.resize(nw * ptr_size + nw * D * realtype_size * 2);
    auto source_ptrs = reinterpret_cast<RealType**>(move_input.data());
    auto new_pos     = reinterpret_cast<RealType*>(move_input.data() + nw * ptr_size);
    auto old_pos     = new_pos + nw * D;

    for (size_t iw = 0; iw < nw; iw++)
    {
      auto& dt             = dt_list.getCastedElement<SoaDistanceTableABOMPTarget>(iw);
      auto& RSoA_OMPTarget = dynamic_cast<const RealSpacePositionsOMPTarget&>(dt.origin_.getCoordinates());
      source_ptrs[iw]      = const_cast<RealType*>(RSoA_OMPTarget.getDevicePtr());
      for (size_t idim = 0; idim < D; idim++)
      {
        new_pos[iw * D + idim] = rnew_list[iw][idim];
        old_pos[iw * D + idim] = p_list[iw].R[iat][idim];
      }
    }

    auto* r_dr_ptr              = mw_new_old_dist_displ.data();
    auto* input_ptr             = move_input.data();
    const int num_sources_local = num_sources_;
    const int nw_local          = nw;

    // nothing on the device reads these unless a consumer has asked for them
    if (temp_data_on_device_)
    {
      ScopedTimer offload(offload_timer_);
      PRAGMA_OFFLOAD("omp target teams distribute parallel for collapse(2)                         map(always, to: input_ptr[:move_input.size()])                         depend(out: r_dr_ptr[:mw_new_old_dist_displ.size()])")
      for (int iw = 0; iw < nw_local; ++iw)
        for (int jat = 0; jat < num_sources_local; ++jat)
        {
          auto* source_pos_ptr = reinterpret_cast<RealType**>(input_ptr)[iw];
          auto* new_pos_ptr    = reinterpret_cast<RealType*>(input_ptr + nw_local * sizeof(RealType*));
          auto* old_pos_ptr    = new_pos_ptr + nw_local * D;

          T pos[D];
          for (int idim = 0; idim < D; idim++)
            pos[idim] = new_pos_ptr[iw * D + idim];
          DTD_BConds<T, D, SC>::computeDistancesOffload(pos, source_pos_ptr, num_padded, r_dr_ptr + iw * stride_size,
                                                        r_dr_ptr + iw * stride_size + num_padded, num_padded, jat);

          if (prepare_old)
          {
            for (int idim = 0; idim < D; idim++)
              pos[idim] = old_pos_ptr[iw * D + idim];
            DTD_BConds<T, D, SC>::computeDistancesOffload(pos, source_pos_ptr, num_padded,
                                                          r_dr_ptr + (iw + nw_local) * stride_size,
                                                          r_dr_ptr + (iw + nw_local) * stride_size + num_padded,
                                                          num_padded, jat);
          }
        }
    }


    /* Host data comes from computing it on the host rather than from bringing the device
     * result back. A single target against all sources is cheap there, which is why the
     * single walker form does it that way, and a copy back would be a blocking transfer
     * on every move whose cost does not fall as walkers per crowd fall. Consumers reading
     * getTempDists() keep exactly what they had; the device copy is an addition.
     */
    for (size_t iw = 0; iw < nw; iw++)
      dt_list[iw].move(p_list[iw], rnew_list[iw], iat, prepare_old);

    /* Last, because the single walker form above clears this: after it, the device side holds
     * this move's distances exactly when the offload above ran.
     */
    dt_leader.temp_data_filled_on_device_ = temp_data_on_device_;
  }

  ///update the stripe for jat-th particle
  inline void update(IndexType iat) override
  {
    std::copy_n(temp_r_.data(), num_sources_, distances_[iat].data());
    for (int idim = 0; idim < D; ++idim)
      std::copy_n(temp_dr_.data(idim), num_sources_, displacements_[iat].data(idim));
  }

private:
  void resize(const size_t num_targets) override
  {
    num_targets_ = num_targets;

    // initialize memory containers and views
    const size_t num_padded  = getAlignedSize<T>(num_sources_);
    const size_t stride_size = getPerTargetPctlStrideSize();
    r_dr_memorypool_.resize(stride_size * num_targets_);

    distances_.resize(num_targets_);
    displacements_.resize(num_targets_);
    for (int i = 0; i < num_targets_; ++i)
    {
      distances_[i].attachReference(r_dr_memorypool_.data() + i * stride_size, num_sources_);
      displacements_[i].attachReference(num_sources_, num_padded,
                                        r_dr_memorypool_.data() + i * stride_size + num_padded);
    }
  }

  /// timer for offload portion
  NewTimer& offload_timer_;
  /** a device side consumer reads the temporary distances of a batch
   *
   * Set by the consumer rather than by configuration, so a run whose components all read
   * the distances on the host leaves the device out of a move entirely.
   */
  mutable bool temp_data_on_device_ = false;
  /// whether the device side of the temp distances describes the move in hand
  mutable bool temp_data_filled_on_device_ = false;
  /// timer for evaluate()
  NewTimer& evaluate_timer_;
};
} // namespace qmcplusplus
#endif
