//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_EEIJASTROW_OPTIMIZED_SOA_H
#define QMCPLUSPLUS_EEIJASTROW_OPTIMIZED_SOA_H
#include <atomic>

#include "Configuration.h"
#if !defined(QMC_BUILD_SANDBOX_ONLY)
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "ResourceCollection.h"
#endif
#include "Particle/DistanceTable.h"
#include "CPU/SIMD/aligned_allocator.hpp"
#include "OMPTarget/OffloadAlignedAllocators.hpp"
#include "CPU/SIMD/algorithm.hpp"
#include <map>
#include <numeric>
#include <memory>

namespace qmcplusplus
{
/** @ingroup WaveFunctionComponent
 *  @brief Specialization for three-body Jastrow function using multiple functors
 *
 *Each pair-type can have distinct function \f$u(r_{ij})\f$.
 *For electrons, distinct pair correlation functions are used
 *for spins up-up/down-down and up-down/down-up.
 */

/** Device-side mirrors of the state JeeIOrbitalSoA::computeU walks on the host.
 *
 * elecs_inside is Array<std::vector<int>,2>: ragged, and a target region cannot follow
 * it. Everything here is the same information in offsets-plus-values form, refreshed
 * per call rather than kept in sync with accepted moves, which keeps the invariant
 * trivial at the price of one repack per ratio evaluation.
 */

template<typename VALT>
struct JeeIMultiWalkerMem : public Resource
{
  Vector<size_t, OffloadPinnedAllocator<size_t>> memb_offsets;
  Vector<int, OffloadPinnedAllocator<int>> memb_elec;
  Vector<VALT, OffloadPinnedAllocator<VALT>> memb_dist;
  /** the displacement of each member from its ion, three components back to back
   *
   * The value of a triplet needs only distances. A gradient or a laplacian needs the
   * displacement of the other electron from the ion as well, so the accept ships it.
   */
  Vector<VALT, OffloadPinnedAllocator<VALT>> memb_displ;
  /** the same entries indexed by the electron they land on rather than by the ion
   *
   * A scatter accumulated with atomics sums in whatever order the threads arrive, so two
   * runs of the same move differ in the last bits and the trajectories part. Grouping the
   * entries by their target lets one thread own an electron and sum its contributions in
   * a fixed order, which is the same arithmetic every time and needs no atomics.
   */
  Vector<int, OffloadPinnedAllocator<int>> memb_ion, memb_grp;
  Vector<size_t, OffloadPinnedAllocator<size_t>> inv_offsets;
  Vector<int, OffloadPinnedAllocator<int>> inv_entry;
  size_t inv_walker_stride = 0;
  Vector<VALT, OffloadPinnedAllocator<VALT>> gamma_flat;
  Vector<char, OffloadPinnedAllocator<char>> fn_have;
  Vector<VALT, OffloadPinnedAllocator<VALT>> ion_cutoff;
  Vector<int, OffloadPinnedAllocator<int>> ion_group;
  /// each electron's spin group, which the accept needs and which never changes
  Vector<int, OffloadPinnedAllocator<int>> elec_group;

  /// pack it once; a walker's electrons do not change group
  void packElecGroups(const ParticleSet& P, int nelec)
  {
    if (elec_group.size() == static_cast<size_t>(nelec))
      return;
    elec_group.resize(nelec);
    for (int jel = 0; jel < nelec; jel++)
      elec_group[jel] = P.GroupID[jel];
    elec_group.updateTo();
  }
  Vector<int, OffloadPinnedAllocator<int>> vp_walker, vp_jg;
  Vector<VALT, OffloadPinnedAllocator<VALT>> vals;
  /** what the accept brings back: per walker the change every other electron sees, and
   * the moved electron's own value, gradient and laplacian before and after the move
   *
   * The change is accumulated as one difference rather than as a new set and an old set,
   * because that is what the accept applies and it halves both the buffer and the atomic
   * traffic that fills it.
   */
  Vector<VALT, OffloadPinnedAllocator<VALT>> acc_delta;
  Vector<VALT, OffloadPinnedAllocator<VALT>> acc_reduce;
  /// per electron partials of the moved electron's own sum, summed in electron order
  Vector<VALT, OffloadPinnedAllocator<VALT>> acc_jpart;
  /// which walker each accepted entry belongs to
  Vector<int, OffloadPinnedAllocator<int>> acc_walker;

  /// membership stamps the device copy was built from, one per walker
  std::vector<size_t> packed_versions;

  size_t memb_walker_stride = 0;
  size_t gamma_size         = 0;
  int N_eI = 0, N_ee = 0, C = 0;
  VALT L = 0;

  JeeIMultiWalkerMem() : Resource("JeeIMultiWalkerMem") {}
  JeeIMultiWalkerMem(const JeeIMultiWalkerMem&) : JeeIMultiWalkerMem() {}
  std::unique_ptr<Resource> makeClone() const override { return std::make_unique<JeeIMultiWalkerMem>(*this); }

  /// flatten every walker's elecs_inside into one offsets/values pair
  template<typename WFCPTRS>
  void packMembership(const WFCPTRS& wfcs, int eGroups, int Nion, int nelec)
  {
    const size_t nw = wfcs.size();
    /* elecs_inside changes only when a move is accepted, and a pass over the quadrature
     * points of one configuration evaluates many ratios without accepting anything, so
     * most calls would rebuild the copy the device already holds. The stamps the last
     * pack read identify that copy.
     */
    bool packed_current = (packed_versions.size() == nw);
    for (size_t iw = 0; packed_current && iw < nw; iw++)
      packed_current = (packed_versions[iw] == wfcs[iw]->getMembershipVersion());
    if (packed_current)
      return;

    packed_versions.resize(nw);
    for (size_t iw = 0; iw < nw; iw++)
      packed_versions[iw] = wfcs[iw]->getMembershipVersion();

    memb_walker_stride = static_cast<size_t>(eGroups) * Nion;
    size_t total       = 0;
    memb_offsets.resize(nw * memb_walker_stride + 1);

    for (size_t iw = 0; iw < nw; iw++)
    {
      const auto& wfc = *wfcs[iw];
      for (int kg = 0; kg < eGroups; kg++)
        for (int iat = 0; iat < Nion; iat++)
        {
          memb_offsets[iw * memb_walker_stride + static_cast<size_t>(kg) * Nion + iat] = total;
          total += wfc.getElecsInside(kg, iat).size();
        }
    }
    memb_offsets[nw * memb_walker_stride] = total;
    memb_elec.resize(total);
    memb_dist.resize(total);
    memb_displ.resize(total * 3);
    memb_ion.resize(total);
    memb_grp.resize(total);

    size_t at = 0;
    for (size_t iw = 0; iw < nw; iw++)
    {
      const auto& wfc = *wfcs[iw];
      for (int kg = 0; kg < eGroups; kg++)
        for (int iat = 0; iat < Nion; iat++)
        {
          const auto& els = wfc.getElecsInside(kg, iat);
          const auto& dst = wfc.getElecsInsideDist(kg, iat);
          const auto& dsp = wfc.getElecsInsideDispl(kg, iat);
          for (size_t n = 0; n < els.size(); n++, at++)
          {
            memb_elec[at] = els[n];
            memb_dist[at] = dst[n];
            memb_ion[at]  = iat;
            memb_grp[at]  = kg;
            for (int idim = 0; idim < 3; idim++)
              memb_displ[at * 3 + idim] = dsp[n][idim];
          }
        }
    }

    /* The same entries again, grouped by the electron they land on. Counting first and
     * filling second keeps each electron's list in increasing entry order, so the sum a
     * thread forms over it does not depend on how the threads were scheduled.
     */
    const int nelec_l = nelec;
    inv_walker_stride = static_cast<size_t>(nelec_l);
    inv_offsets.resize(nw * inv_walker_stride + 1);
    std::vector<size_t> counts(nw * inv_walker_stride, 0);
    for (size_t iw = 0; iw < nw; iw++)
    {
      const size_t begin = memb_offsets[iw * memb_walker_stride];
      const size_t end   = memb_offsets[(iw + 1) * memb_walker_stride];
      for (size_t idx = begin; idx < end; idx++)
        counts[iw * inv_walker_stride + memb_elec[idx]]++;
    }
    size_t running = 0;
    for (size_t slot = 0; slot < nw * inv_walker_stride; slot++)
    {
      inv_offsets[slot] = running;
      running += counts[slot];
    }
    inv_offsets[nw * inv_walker_stride] = running;
    inv_entry.resize(running);
    std::vector<size_t> cursor(inv_offsets.begin(), inv_offsets.end() - 1);
    for (size_t iw = 0; iw < nw; iw++)
    {
      const size_t begin = memb_offsets[iw * memb_walker_stride];
      const size_t end   = memb_offsets[(iw + 1) * memb_walker_stride];
      for (size_t idx = begin; idx < end; idx++)
        inv_entry[cursor[iw * inv_walker_stride + memb_elec[idx]]++] = static_cast<int>(idx);
    }


    memb_offsets.updateTo();
    memb_elec.updateTo();
    memb_dist.updateTo();
    memb_displ.updateTo();
    memb_ion.updateTo();
    memb_grp.updateTo();
    inv_offsets.updateTo();
    inv_entry.updateTo();
  }

  /// one flat gamma block per (ion group, j group, k group), plus a present/absent flag
  template<typename FARRAY>
  void packFunctors(const FARRAY& F, int eGroups, int iGroups)
  {
    /* The flat index below is built from an ion group and two electron groups,
     * so the table has iGroups * eGroups * eGroups entries and not eGroups
     * cubed. Sizing it by the electron count is only large enough while there
     * are no more ion species than electron groups, which holds for a cell of
     * one ion species and two spin channels and not in general. With three ion
     * species and two spin channels the index
     * reaches 11 in a table of 8.
     */
    const size_t ncombo = static_cast<size_t>(iGroups) * eGroups * eGroups;
    fn_have.resize(ncombo);
    std::fill(fn_have.begin(), fn_have.end(), char(0));

    const auto* sample = [&]() -> decltype(F(0, 0, 0)) {
      for (int ig = 0; ig < iGroups; ig++)
        for (int jg = 0; jg < eGroups; jg++)
          for (int kg = 0; kg < eGroups; kg++)
            if (F(ig, jg, kg))
              return F(ig, jg, kg);
      return nullptr;
    }();
    if (sample == nullptr)
      return;

    gamma_size = sample->gammaFlatSize();
    N_eI       = sample->getNeI();
    N_ee       = sample->getNee();
    C          = sample->getC();
    L          = VALT(0.5) * sample->cutoff_radius;

    gamma_flat.resize(gamma_size * ncombo);
    std::fill(gamma_flat.begin(), gamma_flat.end(), VALT(0));
    for (int ig = 0; ig < iGroups; ig++)
      for (int jg = 0; jg < eGroups; jg++)
        for (int kg = 0; kg < eGroups; kg++)
          if (F(ig, jg, kg))
          {
            const size_t fidx = (static_cast<size_t>(ig) * eGroups + jg) * eGroups + kg;
            F(ig, jg, kg)->copyGammaFlat(gamma_flat.data() + fidx * gamma_size);
            fn_have[fidx] = char(1);
          }
    gamma_flat.updateTo();
    fn_have.updateTo();
  }

  template<typename CUTVEC, typename GRPVEC>
  void packIons(const CUTVEC& cutoffs, const GRPVEC& groups, int Nion)
  {
    ion_cutoff.resize(Nion);
    ion_group.resize(Nion);
    for (int iat = 0; iat < Nion; iat++)
    {
      ion_cutoff[iat] = cutoffs[iat];
      ion_group[iat]  = groups[iat];
    }
    ion_cutoff.updateTo();
    ion_group.updateTo();
  }
};

template<class FT>
class JeeIOrbitalSoA : public WaveFunctionComponent
{
  ///type of each component U, dU, d2U;
  using valT = typename FT::real_type;
  ///element position type
  using posT = TinyVector<valT, OHMMS_DIM>;
  ///use the same container
  using DistRow  = DistanceTable::DistRow;
  using DisplRow = DistanceTable::DisplRow;
  ///table index for el-el
  const int ee_Table_ID_;
  ///table index for i-el
  const int ei_Table_ID_;
  //number of particles
  int Nelec, Nion;
  ///number of particles + padded
  size_t Nelec_padded;
  //number of groups of the target particleset
  int eGroups, iGroups;
  ///reference to the sources (ions)
  const ParticleSet& Ions;
  ///diff value
  RealType DiffVal;

  ///\f$Uat[i] = sum_(j) u_{i,j}\f$
  Vector<valT> Uat, oldUk, newUk;
  ///\f$dUat[i] = sum_(j) du_{i,j}\f$
  using gContainer_type = VectorSoaContainer<valT, OHMMS_DIM>;
  gContainer_type dUat, olddUk, newdUk;
  ///\f$d2Uat[i] = sum_(j) d2u_{i,j}\f$
  Vector<valT> d2Uat, oldd2Uk, newd2Uk;
  /// current values during PbyP
  valT cur_Uat, cur_d2Uat;
  posT cur_dUat, dUat_temp;
  ///container for the Jastrow functions
  Array<FT*, 3> F;

  std::map<std::string, std::unique_ptr<FT>> J3Unique;
  //YYYY
  std::map<FT*, int> J3UniqueIndex;
  ///optimizable variables extracted from functors
  OptVariables myVars;

  /// the cutoff for e-I pairs
  std::vector<valT> Ion_cutoff;
  /** stamp identifying the contents of elecs_inside
   *
   * Drawn from a counter that never repeats, so a stamp identifies one state of one
   * object and a consumer holding a copy can tell whether the copy still describes it.
   * A per object counter would let a later object reach a stamp an earlier one had
   * already handed out.
   */
  size_t membership_version_ = 0;
  static inline std::atomic<size_t> membership_stamp_source_{0};
  /// record that elecs_inside no longer matches any copy taken of it
  void touchMembership() { membership_version_ = ++membership_stamp_source_; }

  /// the electrons around ions within the cutoff radius, grouped by species
  Array<std::vector<int>, 2> elecs_inside;
  Array<std::vector<valT>, 2> elecs_inside_dist;
  Array<std::vector<posT>, 2> elecs_inside_displ;
  /// the ids of ions within the cutoff radius of an electron on which a move is proposed
  std::vector<int> ions_nearby_old, ions_nearby_new;

  /// device path for mw_evaluateRatios; off unless the target particle set is offloaded
  bool use_offload_ = false;
  ResourceHandle<JeeIMultiWalkerMem<valT>> mw_mem_handle_;

  /// work buffer size
  size_t Nbuffer;
  /// compressed distances
  aligned_vector<valT> Distjk_Compressed, DistkI_Compressed, DistjI_Compressed;
  std::vector<int> DistIndice_k;
  /// compressed displacements
  gContainer_type Disp_jk_Compressed, Disp_jI_Compressed, Disp_kI_Compressed;
  /// work result buffer
  VectorSoaContainer<valT, 9> mVGL;

  // Used for evaluating derivatives with respect to the parameters
  Array<std::pair<int, int>, 3> VarOffset;
  Vector<RealType> dLogPsi;
  Array<PosType, 2> gradLogPsi;
  Array<RealType, 2> lapLogPsi;

  // Temporary store for parameter derivatives of functor
  // The first index is the functor index in J3Unique.  The second is the parameter index w.r.t. to that
  // functor
  std::vector<std::vector<RealType>> du_dalpha;
  std::vector<std::vector<PosType>> dgrad_dalpha;
  std::vector<std::vector<Tensor<RealType, 3>>> dhess_dalpha;

  void resizeWFOptVectors()
  {
    dLogPsi.resize(myVars.size());
    gradLogPsi.resize(myVars.size(), Nelec);
    lapLogPsi.resize(myVars.size(), Nelec);

    du_dalpha.resize(J3Unique.size());
    dgrad_dalpha.resize(J3Unique.size());
    dhess_dalpha.resize(J3Unique.size());

    int ifunc = 0;
    for (auto& j3UniquePair : J3Unique)
    {
      auto functorPtr           = j3UniquePair.second.get();
      J3UniqueIndex[functorPtr] = ifunc;
      const int numParams       = functorPtr->getNumParameters();
      du_dalpha[ifunc].resize(numParams);
      dgrad_dalpha[ifunc].resize(numParams);
      dhess_dalpha[ifunc].resize(numParams);
      ifunc++;
    }
  }

  /// compute G and L from internally stored data
  QTFull::RealType computeGL(ParticleSet::ParticleGradient& G, ParticleSet::ParticleLaplacian& L) const
  {
    for (int iat = 0; iat < Nelec; ++iat)
    {
      G[iat] += dUat[iat];
      L[iat] += d2Uat[iat];
    }
    return -0.5 * simd::accumulate_n(Uat.data(), Nelec, QTFull::RealType());
  }


public:
  ///alias FuncType
  using FuncType = FT;

  /** @param use_offload take the device ratio path
   *
   * The builder decides this from the deck's gpu attribute, defaulting to the
   * coordinate kind, which is how the one- and two-body Jastrows are told. The device
   * path reads the virtual-particle tables through getMultiWalkerDeviceDataPtr,
   * which only the offload tables provide, so it also needs offload coordinates and
   * is refused here without them.
   */
  JeeIOrbitalSoA(const std::string& obj_name, const ParticleSet& ions, ParticleSet& elecs, bool use_offload = false)
      : WaveFunctionComponent(obj_name),
        /* NEED_VP_FULL_TABLE_ON_HOST is the host path's request, and only its.
         *
         * VirtualParticleSet withholds MW_EVALUATE_RESULT_NO_TRANSFER_TO_HOST from a
         * VP table whenever any component asked for the host copy, so one component
         * asking makes every quadrature evaluation copy the whole multi-walker
         * distance and displacement array back for every consumer of that table. On
         * a 33 ion, 586 electron slab that array is about 11 MB and the copy is a
         * third of the run.
         *
         * The device path does not read it: its kernels name the tables' device
         * addresses. The one- and two-body Jastrows already withdraw the same
         * request when they are offloaded, and on a spline deck this is the only
         * other component that makes it.
         */
        ee_Table_ID_(elecs.addTable(elecs,
                                    DTModes::NEED_TEMP_DATA_ON_HOST |
                                        (wantsOffload(elecs, use_offload) ? DTModes::ALL_OFF
                                                                          : DTModes::NEED_VP_FULL_TABLE_ON_HOST))),
        ei_Table_ID_(elecs.addTable(ions,
                                    DTModes::NEED_FULL_TABLE_ANYTIME |
                                        (wantsOffload(elecs, use_offload) ? DTModes::ALL_OFF
                                                                          : DTModes::NEED_VP_FULL_TABLE_ON_HOST))),
        Ions(ions)
  {
    if (my_name_.empty())
      throw std::runtime_error("JeeIOrbitalSoA object name cannot be empty!");
    use_offload_ = use_offload && elecs.getCoordinates().getKind() == DynamicCoordinateKind::DC_POS_OFFLOAD;
    init(elecs);
  }

  /** the offload decision, in a form the initializer list can use
   *
   * use_offload_ is assigned in the constructor body and the table requests are
   * made before it, so the same test lives here.
   */
  static bool wantsOffload(const ParticleSet& elecs, bool use_offload)
  {
    return use_offload && elecs.getCoordinates().getKind() == DynamicCoordinateKind::DC_POS_OFFLOAD;
  }

  std::string getClassName() const override { return "JeeIOrbitalSoA"; }

  void createResource(ResourceCollection& collection) const override
  {
    collection.addResource(std::make_unique<JeeIMultiWalkerMem<valT>>());
  }

  void acquireResource(ResourceCollection& collection,
                       const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const override
  {
    auto& wfc_leader          = wfc_list.getCastedLeader<JeeIOrbitalSoA<FT>>();
    wfc_leader.mw_mem_handle_ = collection.lendResource<JeeIMultiWalkerMem<valT>>();
  }

  void releaseResource(ResourceCollection& collection,
                       const RefVectorWithLeader<WaveFunctionComponent>& wfc_list) const override
  {
    auto& wfc_leader = wfc_list.getCastedLeader<JeeIOrbitalSoA<FT>>();
    collection.takebackResource(wfc_leader.mw_mem_handle_);
  }

  std::unique_ptr<WaveFunctionComponent> makeClone(ParticleSet& elecs) const override
  {
    auto eeIcopy = std::make_unique<JeeIOrbitalSoA<FT>>(my_name_, Ions, elecs, use_offload_);
    std::map<const FT*, FT*> fcmap;
    for (int iG = 0; iG < iGroups; iG++)
      for (int eG1 = 0; eG1 < eGroups; eG1++)
        for (int eG2 = 0; eG2 < eGroups; eG2++)
        {
          if (F(iG, eG1, eG2) == nullptr)
            continue;
          auto fit = fcmap.find(F(iG, eG1, eG2));
          if (fit == fcmap.end())
          {
            auto fc                = std::make_unique<FT>(*F(iG, eG1, eG2));
            fcmap[F(iG, eG1, eG2)] = fc.get();
            eeIcopy->addFunc(iG, eG1, eG2, std::move(fc));
          }
        }
    // Ye: I don't like the following memory allocated by default.
    eeIcopy->myVars.clear();
    eeIcopy->myVars.insertFrom(myVars);
    eeIcopy->VarOffset = VarOffset;
    return eeIcopy;
  }

  void init(ParticleSet& p)
  {
    Nelec        = p.getTotalNum();
    Nelec_padded = getAlignedSize<valT>(Nelec);
    Nion         = Ions.getTotalNum();
    iGroups      = Ions.getSpeciesSet().getTotalNum();
    eGroups      = p.groups();

    Uat.resize(Nelec);
    dUat.resize(Nelec);
    d2Uat.resize(Nelec);

    oldUk.resize(Nelec);
    olddUk.resize(Nelec);
    oldd2Uk.resize(Nelec);
    newUk.resize(Nelec);
    newdUk.resize(Nelec);
    newd2Uk.resize(Nelec);

    F.resize(iGroups, eGroups, eGroups);
    F = nullptr;
    elecs_inside.resize(eGroups, Nion);
    elecs_inside_dist.resize(eGroups, Nion);
    elecs_inside_displ.resize(eGroups, Nion);
    ions_nearby_old.resize(Nion);
    ions_nearby_new.resize(Nion);
    Ion_cutoff.resize(Nion, 0.0);

    //initialize buffers
    Nbuffer = Nelec;
    mVGL.resize(Nbuffer);
    Distjk_Compressed.resize(Nbuffer);
    DistjI_Compressed.resize(Nbuffer);
    DistkI_Compressed.resize(Nbuffer);
    Disp_jk_Compressed.resize(Nbuffer);
    Disp_jI_Compressed.resize(Nbuffer);
    Disp_kI_Compressed.resize(Nbuffer);
    DistIndice_k.resize(Nbuffer);
  }

  void addFunc(int iSpecies, int eSpecies1, int eSpecies2, std::unique_ptr<FT> j)
  {
    if (eSpecies1 == eSpecies2)
    {
      //if only up-up is specified, assume spin-unpolarized correlations
      if (eSpecies1 == 0)
        for (int eG1 = 0; eG1 < eGroups; eG1++)
          for (int eG2 = 0; eG2 < eGroups; eG2++)
          {
            if (F(iSpecies, eG1, eG2) == 0)
              F(iSpecies, eG1, eG2) = j.get();
          }
    }
    else
    {
      F(iSpecies, eSpecies1, eSpecies2) = j.get();
      F(iSpecies, eSpecies2, eSpecies1) = j.get();
    }
    if (j)
    {
      RealType rcut = 0.5 * j->cutoff_radius;
      for (int i = 0; i < Nion; i++)
        if (Ions.GroupID[i] == iSpecies)
          Ion_cutoff[i] = rcut;
    }
    else
    {
      APP_ABORT("JeeIOrbitalSoA::addFunc  Jastrow function pointer is NULL");
    }
    std::stringstream aname;
    aname << iSpecies << "_" << eSpecies1 << "_" << eSpecies2;
    J3Unique.emplace(aname.str(), std::move(j));
  }


  /** check that correlation information is complete
   */
  void check_complete()
  {
    //check that correlation pointers are either all 0 or all assigned
    bool complete = true;
    for (int i = 0; i < iGroups; ++i)
    {
      int nfilled = 0;
      bool partial;
      for (int e1 = 0; e1 < eGroups; ++e1)
        for (int e2 = 0; e2 < eGroups; ++e2)
          if (F(i, e1, e2) != 0)
            nfilled++;
      partial = nfilled > 0 && nfilled < eGroups * eGroups;
      if (partial)
        app_log() << "J3 eeI is missing correlation for ion " << i << std::endl;
      complete = complete && !partial;
    }
    if (!complete)
    {
      APP_ABORT("JeeIOrbitalSoA::check_complete  J3 eeI is missing correlation components\n  see preceding messages "
                "for details");
    }
    //first set radii
    for (int i = 0; i < Nion; ++i)
    {
      FT* f = F(Ions.GroupID[i], 0, 0);
      if (f != 0)
        Ion_cutoff[i] = .5 * f->cutoff_radius;
    }
    //then check radii
    bool all_radii_match = true;
    for (int i = 0; i < iGroups; ++i)
    {
      if (F(i, 0, 0) != 0)
      {
        bool radii_match = true;
        RealType rcut    = F(i, 0, 0)->cutoff_radius;
        for (int e1 = 0; e1 < eGroups; ++e1)
          for (int e2 = 0; e2 < eGroups; ++e2)
            radii_match = radii_match && F(i, e1, e2)->cutoff_radius == rcut;
        if (!radii_match)
          app_log() << "eeI functors for ion species " << i << " have different radii" << std::endl;
        all_radii_match = all_radii_match && radii_match;
      }
    }
    if (!all_radii_match)
    {
      APP_ABORT("JeeIOrbitalSoA::check_radii  J3 eeI are inconsistent for some ion species\n  see preceding messages "
                "for details");
    }
  }

  bool isOptimizable() const override { return true; }

  void extractOptimizableObjectRefs(UniqueOptObjRefs& opt_obj_refs) override
  {
    for (auto& [key, functor] : J3Unique)
      opt_obj_refs.push_back(*functor);
  }

  /** check out optimizable variables
   */
  void checkOutVariables(const OptVariables& active) override
  {
    myVars.clear();

    for (auto& ftPair : J3Unique)
    {
      ftPair.second->myVars.getIndex(active);
      myVars.insertFrom(ftPair.second->myVars);
    }

    myVars.getIndex(active);
    const size_t NumVars = myVars.size();
    if (NumVars)
    {
      VarOffset.resize(iGroups, eGroups, eGroups);
      int varoffset = myVars.Index[0];
      for (int ig = 0; ig < iGroups; ig++)
        for (int jg = 0; jg < eGroups; jg++)
          for (int kg = 0; kg < eGroups; kg++)
          {
            FT* func_ijk = F(ig, jg, kg);
            if (func_ijk == nullptr)
              continue;
            VarOffset(ig, jg, kg).first  = func_ijk->myVars.Index.front() - varoffset;
            VarOffset(ig, jg, kg).second = func_ijk->myVars.Index.size() + VarOffset(ig, jg, kg).first;
          }
    }
  }

  void build_compact_list(const ParticleSet& P)
  {
    const auto& eI_dists  = P.getDistTableAB(ei_Table_ID_).getDistances();
    const auto& eI_displs = P.getDistTableAB(ei_Table_ID_).getDisplacements();

    for (int iat = 0; iat < Nion; ++iat)
      for (int jg = 0; jg < eGroups; ++jg)
      {
        elecs_inside(jg, iat).clear();
        elecs_inside_dist(jg, iat).clear();
        elecs_inside_displ(jg, iat).clear();
      }

    for (int jg = 0; jg < eGroups; ++jg)
      for (int jel = P.first(jg); jel < P.last(jg); jel++)
        for (int iat = 0; iat < Nion; ++iat)
          if (eI_dists[jel][iat] < Ion_cutoff[iat])
          {
            elecs_inside(jg, iat).push_back(jel);
            elecs_inside_dist(jg, iat).push_back(eI_dists[jel][iat]);
            elecs_inside_displ(jg, iat).push_back(eI_displs[jel][iat]);
          }
    touchMembership();
  }

  LogValue evaluateLog(const ParticleSet& P,
                       ParticleSet::ParticleGradient& G,
                       ParticleSet::ParticleLaplacian& L) override
  {
    recompute(P);
    return log_value_ = computeGL(G, L);
  }

  PsiValue ratio(ParticleSet& P, int iat) override
  {
    UpdateMode = ORB_PBYP_RATIO;

    const auto& eI_table = P.getDistTableAB(ei_Table_ID_);
    const auto& ee_table = P.getDistTableAA(ee_Table_ID_);
    cur_Uat = computeU(P, iat, P.GroupID[iat], eI_table.getTempDists(), ee_table.getTempDists(), ions_nearby_new);
    DiffVal = Uat[iat] - cur_Uat;
    return std::exp(static_cast<PsiValue>(DiffVal));
  }

  /** Batched ratios for the quadrature knots of every walker, on the device.
   *
   * The base class loops walkers serially into evaluateRatios, which loops knots into
   * computeU on the host, one triplet at a time, and the triplet count is the electron
   * count times the ion count times the quadrature knots. The polynomial's own
   * 64-iteration dependency chain is most of the cost of each, and the triplets are
   * mutually independent, which is the case a device is for.
   *
   * The gather runs on the device too. Handing the host the triplets instead would move
   * three doubles for each of them, which on a production slab is hundreds of gigabytes;
   * the filter is cheap in comparison, since only a handful of ions survive the cutoff.
   */
  /// the device path names a reference electron per quadrature point; the host fallback does not
  bool supportsMultiRefRatios() const override { return use_offload_; }

  void mw_evaluateRatios(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                         const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
                         std::vector<std::vector<ValueType>>& ratios) const override
  {
    if (wfc_list.size() == 0)
      return;
    if (!use_offload_)
    {
      WaveFunctionComponent::mw_evaluateRatios(wfc_list, vp_list, ratios);
      return;
    }

    auto& wfc_leader = wfc_list.getCastedLeader<JeeIOrbitalSoA<FT>>();
    auto& vp_leader  = vp_list.getLeader();
    auto& mem        = wfc_leader.mw_mem_handle_.getResource();

    const auto& mw_refPctls = vp_leader.getMultiWalkerRefPctls();
    const size_t nVPs       = mw_refPctls.size();
    const int nw            = wfc_list.size();

    const auto& dt_ei     = vp_leader.getDistTableAB(wfc_leader.ei_Table_ID_);
    const auto& dt_ee     = vp_leader.getDistTableAB(wfc_leader.ee_Table_ID_);
    const RealType* mw_ei = nullptr;
    const RealType* mw_ee = nullptr;
    try
    {
      /* The device address, not the host one.
       *
       * getMultiWalkerDataPtr publishes the host side of a pinned buffer, and a
       * kernel handed that through is_device_ptr does dereference it, across the
       * interconnect, reading whatever the table last transferred back. So a
       * consumer naming the host address is correct only while the table keeps
       * making that transfer, which on a production deck is a third of the run.
       */
      mw_ei = dt_ei.getMultiWalkerDeviceDataPtr();
      mw_ee = dt_ee.getMultiWalkerDeviceDataPtr();
    }
    catch (...)
    {
      WaveFunctionComponent::mw_evaluateRatios(wfc_list, vp_list, ratios);
      return;
    }
    if (mw_ei == nullptr || mw_ee == nullptr)
    {
      WaveFunctionComponent::mw_evaluateRatios(wfc_list, vp_list, ratios);
      return;
    }

    const size_t stride_ei = dt_ei.getPerTargetPctlStrideSize();
    const size_t stride_ee = dt_ee.getPerTargetPctlStrideSize();

    // elecs_inside is Array<std::vector<int>,2>, ragged and host only. Flatten it once
    // per call into offsets plus values, which is also what lets the knots share one
    // walk of it: the host path rebuilds this structure for every knot.
    std::vector<const JeeIOrbitalSoA<FT>*> wfcs(nw);
    for (int iw = 0; iw < nw; iw++)
      wfcs[iw] = &wfc_list.getCastedElement<JeeIOrbitalSoA<FT>>(iw);
    mem.packMembership(wfcs, wfc_leader.eGroups, wfc_leader.Nion, wfc_leader.Nelec);
    mem.packFunctors(wfc_leader.F, wfc_leader.eGroups, wfc_leader.iGroups);
    mem.packIons(wfc_leader.Ion_cutoff, wfc_leader.Ions.GroupID, wfc_leader.Nion);
    mem.vp_walker.resize(nVPs);
    {
      size_t ivp = 0;
      for (int iw = 0; iw < nw; ++iw)
        for (size_t k = 0; k < vp_list[iw].getTotalNum(); ++k, ++ivp)
          mem.vp_walker[ivp] = iw;
      mem.vp_walker.updateTo();
    }
    mem.vals.resize(nVPs);

    const int Nion    = wfc_leader.Nion;
    const int eGroups = wfc_leader.eGroups;
    const int iGroups = wfc_leader.iGroups;
    const auto& refPS = vp_leader.getRefPS();
    mem.vp_jg.resize(nVPs);
    for (size_t ivp = 0; ivp < nVPs; ivp++)
      mem.vp_jg[ivp] = refPS.getGroupID(mw_refPctls[ivp]);
    mem.vp_jg.updateTo();

    auto* memb_off   = mem.memb_offsets.data();
    auto* memb_elec  = mem.memb_elec.data();
    auto* memb_dist  = mem.memb_dist.data();
    auto* gamma_flat = mem.gamma_flat.data();
    auto* fn_have    = mem.fn_have.data();
    auto* ion_cut    = mem.ion_cutoff.data();
    auto* ion_grp    = mem.ion_group.data();
    auto* vals       = mem.vals.data();
    auto* walker_of  = mem.vp_walker.data();
    auto* jg_of      = mem.vp_jg.data();
    auto* refp       = mw_refPctls.data();

    const size_t memb_stride = mem.memb_walker_stride;
    const size_t gsize       = mem.gamma_size;
    const int N_eI_k         = mem.N_eI;
    const int N_ee_k         = mem.N_ee;
    const int C_k            = mem.C;
    const RealType L_k       = mem.L;
    const size_t n_memb      = mem.memb_elec.size();
    const size_t n_off       = mem.memb_offsets.size();

    PRAGMA_OFFLOAD("omp target teams distribute \
                    map(to: refp[:nVPs], walker_of[:nVPs], jg_of[:nVPs]) \
                    map(to: memb_off[:n_off], memb_elec[:n_memb], memb_dist[:n_memb]) \
                    map(to: gamma_flat[:gsize * iGroups * eGroups * eGroups], \
                            fn_have[:iGroups * eGroups * eGroups]) \
                    map(to: ion_cut[:Nion], ion_grp[:Nion]) \
                    map(always, from: vals[:nVPs]) \
                    is_device_ptr(mw_ei, mw_ee)")
    for (size_t ivp = 0; ivp < nVPs; ivp++)
    {
      const int jel          = refp[ivp];
      const int jg           = jg_of[ivp];
      const size_t memb_base = static_cast<size_t>(walker_of[ivp]) * memb_stride;
      const RealType* ei_row = mw_ei + ivp * stride_ei;
      const RealType* ee_row = mw_ee + ivp * stride_ee;
      RealType sum           = 0;

      PRAGMA_OFFLOAD("omp parallel for reduction(+: sum)")
      for (int iat = 0; iat < Nion; iat++)
      {
        const RealType r_jI = ei_row[iat];
        if (r_jI >= ion_cut[iat])
          continue;
        const int ig = ion_grp[iat];
        for (int kg = 0; kg < eGroups; kg++)
        {
          const int fidx = (ig * eGroups + jg) * eGroups + kg;
          if (!fn_have[fidx])
            continue;
          const RealType* grow = gamma_flat + static_cast<size_t>(fidx) * gsize;
          const size_t slot    = memb_base + static_cast<size_t>(kg) * Nion + iat;
          const size_t begin   = memb_off[slot];
          const size_t end     = memb_off[slot + 1];
          for (size_t idx = begin; idx < end; idx++)
          {
            const int kel = memb_elec[idx];
            if (kel == jel)
              continue;
            sum += FT::evaluateV_impl(ee_row[kel], r_jI, memb_dist[idx], grow, N_eI_k, N_ee_k, C_k, L_k);
          }
        }
      }
      vals[ivp] = sum;
    }

    size_t ivp = 0;
    for (int iw = 0; iw < nw; ++iw)
    {
      const auto& wfc = wfc_list.getCastedElement<JeeIOrbitalSoA<FT>>(iw);
      for (size_t k = 0; k < vp_list[iw].getTotalNum(); ++k, ++ivp)
        ratios[iw][k] = std::exp(wfc.Uat[mw_refPctls[ivp]] - mem.vals[ivp]);
    }
    assert(ivp == nVPs);
  }

  const std::vector<int>& getElecsInside(int kg, int iat) const { return elecs_inside(kg, iat); }
  size_t getMembershipVersion() const { return membership_version_; }
  const std::vector<valT>& getElecsInsideDist(int kg, int iat) const { return elecs_inside_dist(kg, iat); }
  const std::vector<posT>& getElecsInsideDispl(int kg, int iat) const { return elecs_inside_displ(kg, iat); }

  void evaluateRatios(const VirtualParticleSet& VP, std::vector<ValueType>& ratios) override
  {
    assert(VP.getTotalNum() == ratios.size());
    for (int k = 0; k < ratios.size(); ++k)
      ratios[k] = std::exp(Uat[VP.refPtcl] -
                           computeU(VP.getRefPS(), VP.refPtcl, VP.getRefPS().GroupID[VP.refPtcl],
                                    VP.getDistTableAB(ei_Table_ID_).getDistRow(k),
                                    VP.getDistTableAB(ee_Table_ID_).getDistRow(k), ions_nearby_old));
  }

  void evaluateRatiosAlltoOne(ParticleSet& P, std::vector<ValueType>& ratios) override
  {
    const auto& eI_table = P.getDistTableAB(ei_Table_ID_);
    const auto& eI_dists = eI_table.getDistances();
    const auto& ee_table = P.getDistTableAA(ee_Table_ID_);

    for (int jg = 0; jg < eGroups; ++jg)
    {
      const valT sumU = computeU(P, -1, jg, eI_table.getTempDists(), ee_table.getTempDists(), ions_nearby_new);

      for (int j = P.first(jg); j < P.last(jg); ++j)
      {
        // remove self-interaction
        valT Uself(0);
        for (int iat = 0; iat < Nion; ++iat)
        {
          const valT& r_Ij = eI_table.getTempDists()[iat];
          const valT& r_Ik = eI_dists[j][iat];
          if (r_Ij < Ion_cutoff[iat] && r_Ik < Ion_cutoff[iat])
          {
            const int ig = Ions.GroupID[iat];
            Uself += F(ig, jg, jg)->evaluate(ee_table.getTempDists()[j], r_Ij, r_Ik);
          }
        }
        ratios[j] = std::exp(Uat[j] + Uself - sumU);
      }
    }
  }

  GradType evalGrad(ParticleSet& P, int iat) override { return GradType(dUat[iat]); }

  PsiValue ratioGrad(ParticleSet& P, int iat, GradType& grad_iat) override
  {
    UpdateMode = ORB_PBYP_PARTIAL;

    const auto& eI_table = P.getDistTableAB(ei_Table_ID_);
    const auto& ee_table = P.getDistTableAA(ee_Table_ID_);
    computeU3(P, iat, eI_table.getTempDists(), eI_table.getTempDispls(), ee_table.getTempDists(),
              ee_table.getTempDispls(), cur_Uat, cur_dUat, cur_d2Uat, newUk, newdUk, newd2Uk, ions_nearby_new);
    DiffVal = Uat[iat] - cur_Uat;
    grad_iat += cur_dUat;
    return std::exp(static_cast<PsiValue>(DiffVal));
  }

  inline void restore(int iat) override {}

  /** the accept for a batch, with the triplet sums and the scatter done on the device
   *
   * The single walker accept runs computeU3 twice, once for the position left and once
   * for the one taken, and each walks the triplets the moved electron makes with the ions
   * near it and the electrons near those ions. Both produce a sum for the moved electron
   * and a scatter into every other electron of a triplet, and the scatter is why this
   * could not follow the one body path: a reduction alone leaves the accept recomputing
   * what it needs, which at a high acceptance is most of the work.
   *
   * The two passes accumulate one difference rather than two sets, since the accept only
   * ever applies new minus old. What comes back is that difference plus the moved
   * electron's own value, gradient and laplacian on both sides, and the host applies it.
   */
  void mw_accept_rejectMove(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                            const RefVectorWithLeader<ParticleSet>& p_list,
                            int iat,
                            const std::vector<bool>& isAccepted,
                            bool safe_to_delay = false) const override
  {
    auto& wfc_leader = wfc_list.getCastedLeader<JeeIOrbitalSoA<FT>>();
    const int nw     = wfc_list.size();

    std::vector<int> accepted;
    accepted.reserve(nw);
    for (int iw = 0; iw < nw; iw++)
      if (isAccepted[iw])
        accepted.push_back(iw);

    auto& p_leader        = p_list.getLeader();
    const auto& dt_ei     = p_leader.getDistTableAB(wfc_leader.ei_Table_ID_);
    const auto& dt_ee     = p_leader.getDistTableAA(wfc_leader.ee_Table_ID_);
    const bool worthwhile = use_offload_ && accepted.size() > 1;

    /* The electron ion table forms its temporary distances on the device for a consumer
     * that has asked, and it forms them on the next move rather than this one. Announcing
     * the need and then asking whether it is met reads back the flag just set and walks
     * into a buffer nothing has written, so the batch that announces takes the host path.
     */
    bool device_ready = false;
    if (worthwhile)
    {
      device_ready = dt_ei.hasTempDataOnDevice();
      if (!device_ready)
        dt_ei.requireTempDataOnDevice();
    }

    if (!device_ready)
    {
      WaveFunctionComponent::mw_accept_rejectMove(wfc_list, p_list, iat, isAccepted, safe_to_delay);
      return;
    }

    const RealType* mw_ei = nullptr;
    const RealType* mw_ee = nullptr;
    try
    {
      // this table writes only the device side of its temporary buffer, so name that
      mw_ei = dt_ei.getMultiWalkerTempDeviceDataPtr();
      mw_ee = dt_ee.getMultiWalkerTempDeviceDataPtr();
    }
    catch (...)
    {
      WaveFunctionComponent::mw_accept_rejectMove(wfc_list, p_list, iat, isAccepted, safe_to_delay);
      return;
    }
    if (mw_ei == nullptr || mw_ee == nullptr)
    {
      WaveFunctionComponent::mw_accept_rejectMove(wfc_list, p_list, iat, isAccepted, safe_to_delay);
      return;
    }

    auto& mem = wfc_leader.mw_mem_handle_.getResource();

    /* The accept does not need the packed membership at all.
     *
     * It exists to tell the kernel which electrons are inside which ions, and the
     * electron-ion table is already on the device: the constructor asks for
     * NEED_FULL_TABLE_ANYTIME. So the kernel reads that table and tests the ion
     * cutoff itself, which is one comparison per ion against walking a packed list
     * that a host to device transfer has to deliver first.
     */
    const RealType* mw_ei_full = nullptr;
    size_t ei_full_stride      = 0;
    try
    {
      mw_ei_full     = dt_ei.getMultiWalkerDeviceDataPtr();
      ei_full_stride = dt_ei.getPerTargetPctlStrideSize();
    }
    catch (...)
    {
      WaveFunctionComponent::mw_accept_rejectMove(wfc_list, p_list, iat, isAccepted, safe_to_delay);
      return;
    }
    if (mw_ei_full == nullptr || ei_full_stride == 0)
    {
      WaveFunctionComponent::mw_accept_rejectMove(wfc_list, p_list, iat, isAccepted, safe_to_delay);
      return;
    }

    // the membership still describes the configuration before any of these moves lands
    std::vector<const JeeIOrbitalSoA<FT>*> wfcs(nw);
    for (int iw = 0; iw < nw; iw++)
      wfcs[iw] = &wfc_list.getCastedElement<JeeIOrbitalSoA<FT>>(iw);
    mem.packElecGroups(p_leader, wfc_leader.Nelec);
    mem.packFunctors(wfc_leader.F, wfc_leader.eGroups, wfc_leader.iGroups);
    mem.packIons(wfc_leader.Ion_cutoff, wfc_leader.Ions.GroupID, wfc_leader.Nion);

    const int na        = accepted.size();
    const int Nelec_l   = wfc_leader.Nelec;
    const int Nion      = wfc_leader.Nion;
    const int eGroups   = wfc_leader.eGroups;
    const int iGroups   = wfc_leader.iGroups;
    const int jg        = p_leader.GroupID[iat];
    const size_t nfield = 5;

    mem.acc_walker.resize(na);
    for (int ia = 0; ia < na; ia++)
      mem.acc_walker[ia] = accepted[ia];
    mem.acc_walker.updateTo();
    mem.acc_delta.resize(static_cast<size_t>(na) * nfield * Nelec_l);
    mem.acc_reduce.resize(static_cast<size_t>(na) * 10);
    mem.acc_jpart.resize(static_cast<size_t>(na) * nfield * Nelec_l);

    auto* elec_grp           = mem.elec_group.data();
    const size_t ei_stride   = ei_full_stride;
    const size_t pad_ei_full = getAlignedSize<RealType>(wfc_leader.Nion);
    auto* memb_off           = mem.memb_offsets.data();
    auto* memb_elec          = mem.memb_elec.data();
    auto* memb_dist          = mem.memb_dist.data();
    auto* memb_displ         = mem.memb_displ.data();
    auto* gamma_flat         = mem.gamma_flat.data();
    auto* fn_have            = mem.fn_have.data();
    auto* ion_cut            = mem.ion_cutoff.data();
    auto* ion_grp            = mem.ion_group.data();
    auto* delta_ptr          = mem.acc_delta.data();
    auto* reduce_ptr         = mem.acc_reduce.data();
    auto* walker_ptr         = mem.acc_walker.data();
    auto* jpart_ptr          = mem.acc_jpart.data();
    auto* memb_ion           = mem.memb_ion.data();
    auto* memb_grp           = mem.memb_grp.data();
    auto* inv_off            = mem.inv_offsets.data();
    auto* inv_ent            = mem.inv_entry.data();

    const size_t memb_stride = mem.memb_walker_stride;
    const size_t gsize       = mem.gamma_size;
    const int N_eI_k         = mem.N_eI;
    const int N_ee_k         = mem.N_ee;
    const int C_k            = mem.C;
    const RealType L_k       = mem.L;
    const size_t n_memb      = mem.memb_elec.size();
    const size_t n_off       = mem.memb_offsets.size();
    /* Both tables lay a walker's temporary distances out as the padded count followed by
     * the displacement components, the position taken first and the one left after every
     * walker's. The electron electron table does not publish the stride, so it is formed
     * the same way here for both.
     */
    const size_t pad_ei     = getAlignedSize<RealType>(Nion);
    const size_t pad_ee     = getAlignedSize<RealType>(Nelec_l);
    const size_t stride_ei  = pad_ei * (OHMMS_DIM + 1);
    const size_t stride_ee  = pad_ee * (OHMMS_DIM + 1);
    const size_t delta_len  = mem.acc_delta.size();
    const size_t inv_stride = mem.inv_walker_stride;
    const size_t n_inv_off  = mem.inv_offsets.size();
    const size_t n_inv_ent  = mem.inv_entry.size();
    constexpr RealType lapfac(OHMMS_DIM - 1);

    PRAGMA_OFFLOAD("omp target teams distribute num_teams(na) \
                      map(to: walker_ptr[:na]) \
                      map(to: elec_grp[:Nelec_l]) \
                      map(alloc: jpart_ptr[:delta_len]) \
                      map(to: gamma_flat[:gsize * iGroups * eGroups * eGroups], \
                              fn_have[:iGroups * eGroups * eGroups]) \
                      map(to: ion_cut[:Nion], ion_grp[:Nion]) \
                      map(always, from: delta_ptr[:delta_len], reduce_ptr[:na * 10]) \
                      is_device_ptr(mw_ei, mw_ee, mw_ei_full)")
    for (int ia = 0; ia < na; ia++)
    {
      const int iw            = walker_ptr[ia];
      auto* restrict delta_iw = delta_ptr + static_cast<size_t>(ia) * nfield * Nelec_l;
      auto* restrict jpart_iw = jpart_ptr + static_cast<size_t>(ia) * nfield * Nelec_l;

      PRAGMA_OFFLOAD("omp parallel for")
      for (int k = 0; k < static_cast<int>(nfield) * Nelec_l; k++)
        delta_iw[k] = RealType(0);

      for (int pass = 0; pass < 2; pass++)
      {
        // pass 0 is the position taken, pass 1 the one left; the accept applies the
        // difference, so the second enters with the opposite sign
        const RealType sign     = pass == 0 ? RealType(1) : RealType(-1);
        const size_t ei_base    = (pass == 0 ? iw : iw + nw) * stride_ei;
        const size_t ee_base    = (pass == 0 ? iw : iw + nw) * stride_ee;
        const RealType* ei_dist = mw_ei + ei_base;
        const RealType* ei_disp = ei_dist + pad_ei;
        const RealType* ee_dist = mw_ee + ee_base;
        const RealType* ee_disp = ee_dist + pad_ee;

        /* One thread per electron, each summing the entries that land on it in the
           * order the pack laid them down. The moved electron's own sum is gathered the
           * same way, as a partial per electron, and added up afterwards in electron
           * order. Nothing here depends on how the threads were scheduled.
           */
        PRAGMA_OFFLOAD("omp parallel for")
        for (int kel = 0; kel < Nelec_l; kel++)
        {
          RealType jU = 0, jG0 = 0, jG1 = 0, jG2 = 0, jL = 0;
          RealType kU = 0, kG0 = 0, kG1 = 0, kG2 = 0, kL = 0;

          if (kel != iat)
          {
            /* Which ions this electron is inside is a comparison, not a lookup.
             * The electron-ion table is on the device, so the row for (iw, kel)
             * gives both the distance and the displacement, and the cutoff test is
             * the same one the host membership applied when it built the lists: one
             * comparison per ion, and no transfer.
             */
            const int kg                    = elec_grp[kel];
            const RealType* restrict kI_row = mw_ei_full + (static_cast<size_t>(iw) * Nelec_l + kel) * ei_stride;
            for (int iat_ion = 0; iat_ion < Nion; iat_ion++)
            {
              const RealType r_jI = ei_dist[iat_ion];
              if (r_jI >= ion_cut[iat_ion])
                continue;
              const RealType r_kI_test = kI_row[iat_ion];
              if (r_kI_test >= ion_cut[iat_ion])
                continue;
              const int fidx = (ion_grp[iat_ion] * eGroups + jg) * eGroups + kg;
              if (!fn_have[fidx])
                continue;

              const RealType* grow = gamma_flat + static_cast<size_t>(fidx) * gsize;
              const RealType jI0   = ei_disp[iat_ion];
              const RealType jI1   = ei_disp[pad_ei + iat_ion];
              const RealType jI2   = ei_disp[2 * pad_ei + iat_ion];
              const RealType r_kI  = r_kI_test;
              const RealType r_jk  = ee_dist[kel];
              const RealType jk0   = ee_disp[kel];
              const RealType jk1   = ee_disp[pad_ee + kel];
              const RealType jk2   = ee_disp[2 * pad_ee + kel];
              const RealType kI0   = kI_row[pad_ei_full + iat_ion];
              const RealType kI1   = kI_row[2 * pad_ei_full + iat_ion];
              const RealType kI2   = kI_row[3 * pad_ei_full + iat_ion];

              RealType val, g0, g1, g2, h00, h01, h02, h11, h22;
              FT::evaluateVGH_impl(r_jk, r_jI, r_kI, grow, N_eI_k, N_ee_k, C_k, L_k, val, g0, g1, g2, h00, h01, h02,
                                   h11, h22);

              const RealType dot_jk_jI = jk0 * jI0 + jk1 * jI1 + jk2 * jI2;
              const RealType dot_kI_jk = kI0 * jk0 + kI1 * jk1 + kI2 * jk2;

              jU += val;
              jG0 += g1 * jI0 + g0 * jk0;
              jG1 += g1 * jI1 + g0 * jk1;
              jG2 += g1 * jI2 + g0 * jk2;
              jL -= h00 + h11 + lapfac * (g0 + g1) + RealType(2) * h01 * dot_jk_jI;

              kU += val;
              kG0 += g2 * kI0 - g0 * jk0;
              kG1 += g2 * kI1 - g0 * jk1;
              kG2 += g2 * kI2 - g0 * jk2;
              kL -= h00 + h22 + lapfac * (g0 + g2) - RealType(2) * h02 * dot_kI_jk;
            }
          }

          delta_iw[kel] += sign * kU;
          delta_iw[Nelec_l + kel] += sign * kG0;
          delta_iw[2 * Nelec_l + kel] += sign * kG1;
          delta_iw[3 * Nelec_l + kel] += sign * kG2;
          delta_iw[4 * Nelec_l + kel] += sign * kL;

          jpart_iw[kel]               = jU;
          jpart_iw[Nelec_l + kel]     = jG0;
          jpart_iw[2 * Nelec_l + kel] = jG1;
          jpart_iw[3 * Nelec_l + kel] = jG2;
          jpart_iw[4 * Nelec_l + kel] = jL;
        }

        // and the moved electron's own sum, added in electron order
        RealType Uj = 0, dUj0 = 0, dUj1 = 0, dUj2 = 0, d2Uj = 0;
        for (int kel = 0; kel < Nelec_l; kel++)
        {
          Uj += jpart_iw[kel];
          dUj0 += jpart_iw[Nelec_l + kel];
          dUj1 += jpart_iw[2 * Nelec_l + kel];
          dUj2 += jpart_iw[3 * Nelec_l + kel];
          d2Uj += jpart_iw[4 * Nelec_l + kel];
        }

        const int off       = ia * 10 + pass * 5;
        reduce_ptr[off]     = Uj;
        reduce_ptr[off + 1] = dUj0;
        reduce_ptr[off + 2] = dUj1;
        reduce_ptr[off + 3] = dUj2;
        reduce_ptr[off + 4] = d2Uj;
      }
    }

    // apply what came back, then let the per walker accept finish the bookkeeping
    for (int ia = 0; ia < na; ia++)
    {
      auto& wfc              = wfc_list.getCastedElement<JeeIOrbitalSoA<FT>>(accepted[ia]);
      const RealType* delta  = mem.acc_delta.data() + static_cast<size_t>(ia) * nfield * Nelec_l;
      const RealType* reduce = mem.acc_reduce.data() + ia * 10;

      for (int jel = 0; jel < Nelec_l; jel++)
      {
        wfc.Uat[jel] += delta[jel];
        wfc.d2Uat[jel] += delta[4 * Nelec_l + jel];
      }
      for (int idim = 0; idim < OHMMS_DIM; idim++)
      {
        valT* restrict save_g = wfc.dUat.data(idim);
        for (int jel = 0; jel < Nelec_l; jel++)
          save_g[jel] += delta[(1 + idim) * Nelec_l + jel];
      }

      const RealType new_Uj = reduce[0];
      const RealType old_Uj = reduce[5];
      wfc.log_value_ += old_Uj - new_Uj;
      wfc.Uat[iat]   = new_Uj;
      wfc.dUat(iat)  = posT(reduce[1], reduce[2], reduce[3]);
      wfc.d2Uat[iat] = reduce[4];

      // the membership update stays here: it is a handful of host side list edits
      wfc.updateMembershipAfterAccept(p_list[accepted[ia]], iat);
    }
  }

  void acceptMove(ParticleSet& P, int iat, bool safe_to_delay = false) override
  {
    const auto& eI_table = P.getDistTableAB(ei_Table_ID_);
    const auto& ee_table = P.getDistTableAA(ee_Table_ID_);
    // get the old value, grad, lapl
    computeU3(P, iat, eI_table.getDistRow(iat), eI_table.getDisplRow(iat), ee_table.getOldDists(),
              ee_table.getOldDispls(), Uat[iat], dUat_temp, d2Uat[iat], oldUk, olddUk, oldd2Uk, ions_nearby_old);
    if (UpdateMode == ORB_PBYP_RATIO)
    { //ratio-only during the move; need to compute derivatives
      computeU3(P, iat, eI_table.getTempDists(), eI_table.getTempDispls(), ee_table.getTempDists(),
                ee_table.getTempDispls(), cur_Uat, cur_dUat, cur_d2Uat, newUk, newdUk, newd2Uk, ions_nearby_new);
    }

#pragma omp simd
    for (int jel = 0; jel < Nelec; jel++)
    {
      Uat[jel] += newUk[jel] - oldUk[jel];
      d2Uat[jel] += newd2Uk[jel] - oldd2Uk[jel];
    }
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
    {
      valT* restrict save_g      = dUat.data(idim);
      const valT* restrict new_g = newdUk.data(idim);
      const valT* restrict old_g = olddUk.data(idim);
#pragma omp simd aligned(save_g, new_g, old_g : QMC_SIMD_ALIGNMENT)
      for (int jel = 0; jel < Nelec; jel++)
        save_g[jel] += new_g[jel] - old_g[jel];
    }

    log_value_ += Uat[iat] - cur_Uat;
    Uat[iat]   = cur_Uat;
    dUat(iat)  = cur_dUat;
    d2Uat[iat] = cur_d2Uat;

    updateMembershipAfterAccept(P, iat);
  }

  /** move the moved electron between the per ion member lists
   *
   * Extracted so the batched accept can reuse it. computeU3 leaves the two ion lists
   * behind as a side effect of walking the triplets; a form that does the walking on
   * the device has to rebuild them, and the test is the one computeU3 applies.
   */
  void updateMembershipAfterAccept(const ParticleSet& P, int iat)
  {
    const auto& eI_table = P.getDistTableAB(ei_Table_ID_);
    ions_nearby_old.clear();
    ions_nearby_new.clear();
    for (int jat = 0; jat < Nion; jat++)
    {
      if (eI_table.getDistRow(iat)[jat] < Ion_cutoff[jat])
        ions_nearby_old.push_back(jat);
      if (eI_table.getTempDists()[jat] < Ion_cutoff[jat])
        ions_nearby_new.push_back(jat);
    }

    const int ig = P.GroupID[iat];
    // update compact list elecs_inside
    touchMembership();
    // if the old position exists in elecs_inside
    for (int iind = 0; iind < ions_nearby_old.size(); iind++)
    {
      int jat         = ions_nearby_old[iind];
      auto iter       = std::find(elecs_inside(ig, jat).begin(), elecs_inside(ig, jat).end(), iat);
      auto iter_dist  = elecs_inside_dist(ig, jat).begin() + std::distance(elecs_inside(ig, jat).begin(), iter);
      auto iter_displ = elecs_inside_displ(ig, jat).begin() + std::distance(elecs_inside(ig, jat).begin(), iter);
      // If not found, segfault can happen later. Stop here.
      if (iter == elecs_inside(ig, jat).end())
      {
        std::ostringstream msg;
        msg << "Report bug! Updating electron iat = " << iat << " near ion " << jat
            << " distance = " << std::setprecision(std::numeric_limits<float>::digits10 + 1)
            << eI_table.getDistRow(iat)[jat] << ". Failed to find it in elecs_inside!" << std::endl;
        throw std::runtime_error(msg.str());
      }
#ifndef NDEBUG
      else if (std::abs(eI_table.getDistRow(iat)[jat] - *iter_dist) >= 10 * std::numeric_limits<float>::epsilon())
      {
        std::ostringstream msg;
        msg << "Report bug! Inconsistent electron iat = " << iat << " near ion " << jat << " dist "
            << std::setprecision(std::numeric_limits<float>::digits10 + 1) << eI_table.getDistRow(iat)[jat]
            << " stored value = " << *iter_dist
            << ". eI distance stored value elecs_inside_dist does not match distance table!" << std::endl;
        throw std::runtime_error(msg.str());
      }
#endif

      if (eI_table.getTempDists()[jat] < Ion_cutoff[jat]) // the new position is still inside
      {
        *iter_dist                                                      = eI_table.getTempDists()[jat];
        *iter_displ                                                     = eI_table.getTempDispls()[jat];
        *std::find(ions_nearby_new.begin(), ions_nearby_new.end(), jat) = -1;
      }
      else
      {
        *iter = elecs_inside(ig, jat).back();
        elecs_inside(ig, jat).pop_back();
        *iter_dist = elecs_inside_dist(ig, jat).back();
        elecs_inside_dist(ig, jat).pop_back();
        *iter_displ = elecs_inside_displ(ig, jat).back();
        elecs_inside_displ(ig, jat).pop_back();
      }
    }

    // if the old position doesn't exist in elecs_inside but the new position do
    for (int iind = 0; iind < ions_nearby_new.size(); iind++)
    {
      int jat = ions_nearby_new[iind];
      if (jat >= 0)
      {
        elecs_inside(ig, jat).push_back(iat);
        elecs_inside_dist(ig, jat).push_back(eI_table.getTempDists()[jat]);
        elecs_inside_displ(ig, jat).push_back(eI_table.getTempDispls()[jat]);
      }
    }
  }

  inline void recompute(const ParticleSet& P) override
  {
    const auto& eI_table = P.getDistTableAB(ei_Table_ID_);
    const auto& ee_table = P.getDistTableAA(ee_Table_ID_);

    build_compact_list(P);

    for (int jel = 0; jel < Nelec; ++jel)
    {
      computeU3(P, jel, eI_table.getDistRow(jel), eI_table.getDisplRow(jel), ee_table.getDistRow(jel),
                ee_table.getDisplRow(jel), Uat[jel], dUat_temp, d2Uat[jel], newUk, newdUk, newd2Uk, ions_nearby_new,
                true);
      dUat(jel) = dUat_temp;
// add the contribution from the upper triangle
#pragma omp simd
      for (int kel = 0; kel < jel; kel++)
      {
        Uat[kel] += newUk[kel];
        d2Uat[kel] += newd2Uk[kel];
      }
      for (int idim = 0; idim < OHMMS_DIM; ++idim)
      {
        valT* restrict save_g      = dUat.data(idim);
        const valT* restrict new_g = newdUk.data(idim);
#pragma omp simd aligned(save_g, new_g : QMC_SIMD_ALIGNMENT)
        for (int kel = 0; kel < jel; kel++)
          save_g[kel] += new_g[kel];
      }
    }
  }

  void mw_recompute(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                    const RefVectorWithLeader<ParticleSet>& p_list,
                    const std::vector<bool>& recompute) const override
  {
    for (int iw = 0; iw < wfc_list.size(); iw++)
      if (auto& jeei = wfc_list.getCastedElement<JeeIOrbitalSoA>(iw); recompute[iw])
        jeei.recompute(p_list[iw]);
      else
        // distance values may change due to recomputing and thus bring internal data up-to-date
        jeei.build_compact_list(p_list[iw]);
  }

  inline valT computeU(const ParticleSet& P,
                       int jel,
                       int jg,
                       const DistRow& distjI,
                       const DistRow& distjk,
                       std::vector<int>& ions_nearby)
  {
    ions_nearby.clear();
    for (int iat = 0; iat < Nion; ++iat)
      if (distjI[iat] < Ion_cutoff[iat])
        ions_nearby.push_back(iat);

    valT Uj = valT(0);
    for (int kg = 0; kg < eGroups; ++kg)
    {
      int kel_counter = 0;
      for (int iind = 0; iind < ions_nearby.size(); ++iind)
      {
        const int iat   = ions_nearby[iind];
        const int ig    = Ions.GroupID[iat];
        const valT r_jI = distjI[iat];
        for (int kind = 0; kind < elecs_inside(kg, iat).size(); kind++)
        {
          const int kel = elecs_inside(kg, iat)[kind];
          if (kel != jel)
          {
            DistkI_Compressed[kel_counter] = elecs_inside_dist(kg, iat)[kind];
            Distjk_Compressed[kel_counter] = distjk[kel];
            DistjI_Compressed[kel_counter] = r_jI;
            kel_counter++;
            if (kel_counter == Nbuffer)
            {
              const FT& feeI(*F(ig, jg, kg));
              Uj += feeI.evaluateV(kel_counter, Distjk_Compressed.data(), DistjI_Compressed.data(),
                                   DistkI_Compressed.data());
              kel_counter = 0;
            }
          }
        }
        if ((iind + 1 == ions_nearby.size() || ig != Ions.GroupID[ions_nearby[iind + 1]]) && kel_counter > 0)
        {
          const FT& feeI(*F(ig, jg, kg));
          Uj +=
              feeI.evaluateV(kel_counter, Distjk_Compressed.data(), DistjI_Compressed.data(), DistkI_Compressed.data());
          kel_counter = 0;
        }
      }
    }
    return Uj;
  }

  inline void computeU3_engine(const ParticleSet& P,
                               const FT& feeI,
                               int kel_counter,
                               valT& Uj,
                               posT& dUj,
                               valT& d2Uj,
                               Vector<valT>& Uk,
                               gContainer_type& dUk,
                               Vector<valT>& d2Uk)
  {
    constexpr valT czero(0);
    constexpr valT cone(1);
    constexpr valT ctwo(2);
    constexpr valT lapfac = OHMMS_DIM - cone;

    valT* restrict val     = mVGL.data(0);
    valT* restrict gradF0  = mVGL.data(1);
    valT* restrict gradF1  = mVGL.data(2);
    valT* restrict gradF2  = mVGL.data(3);
    valT* restrict hessF00 = mVGL.data(4);
    valT* restrict hessF11 = mVGL.data(5);
    valT* restrict hessF22 = mVGL.data(6);
    valT* restrict hessF01 = mVGL.data(7);
    valT* restrict hessF02 = mVGL.data(8);

    feeI.evaluateVGL(kel_counter, Distjk_Compressed.data(), DistjI_Compressed.data(), DistkI_Compressed.data(), val,
                     gradF0, gradF1, gradF2, hessF00, hessF11, hessF22, hessF01, hessF02);

    // compute the contribution to jel, kel
    Uj               = simd::accumulate_n(val, kel_counter, Uj);
    valT gradF0_sum  = simd::accumulate_n(gradF0, kel_counter, czero);
    valT gradF1_sum  = simd::accumulate_n(gradF1, kel_counter, czero);
    valT hessF00_sum = simd::accumulate_n(hessF00, kel_counter, czero);
    valT hessF11_sum = simd::accumulate_n(hessF11, kel_counter, czero);
    d2Uj -= hessF00_sum + hessF11_sum + lapfac * (gradF0_sum + gradF1_sum);
    std::fill_n(hessF11, kel_counter, czero);
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
    {
      valT* restrict jk = Disp_jk_Compressed.data(idim);
      valT* restrict jI = Disp_jI_Compressed.data(idim);
      valT* restrict kI = Disp_kI_Compressed.data(idim);
      valT dUj_x(0);
#pragma omp simd aligned(gradF0, gradF1, gradF2, hessF11, jk, jI, kI : QMC_SIMD_ALIGNMENT) reduction(+ : dUj_x)
      for (int kel_index = 0; kel_index < kel_counter; kel_index++)
      {
        // recycle hessF11
        hessF11[kel_index] += kI[kel_index] * jk[kel_index];
        dUj_x += gradF1[kel_index] * jI[kel_index];
        // destroy jk, kI
        const valT temp = jk[kel_index] * gradF0[kel_index];
        dUj_x += temp;
        jk[kel_index] *= jI[kel_index];
        kI[kel_index] = kI[kel_index] * gradF2[kel_index] - temp;
      }
      dUj[idim] += dUj_x;

      valT* restrict jk0 = Disp_jk_Compressed.data(0);
      if (idim > 0)
      {
#pragma omp simd aligned(jk, jk0 : QMC_SIMD_ALIGNMENT)
        for (int kel_index = 0; kel_index < kel_counter; kel_index++)
          jk0[kel_index] += jk[kel_index];
      }

      valT* restrict dUk_x = dUk.data(idim);
      for (int kel_index = 0; kel_index < kel_counter; kel_index++)
        dUk_x[DistIndice_k[kel_index]] += kI[kel_index];
    }
    valT sum(0);
    valT* restrict jk0 = Disp_jk_Compressed.data(0);
#pragma omp simd aligned(jk0, hessF01 : QMC_SIMD_ALIGNMENT) reduction(+ : sum)
    for (int kel_index = 0; kel_index < kel_counter; kel_index++)
      sum += hessF01[kel_index] * jk0[kel_index];
    d2Uj -= ctwo * sum;

#pragma omp simd aligned(hessF00, hessF22, gradF0, gradF2, hessF02, hessF11 : QMC_SIMD_ALIGNMENT)
    for (int kel_index = 0; kel_index < kel_counter; kel_index++)
      hessF00[kel_index] = hessF00[kel_index] + hessF22[kel_index] + lapfac * (gradF0[kel_index] + gradF2[kel_index]) -
          ctwo * hessF02[kel_index] * hessF11[kel_index];

    for (int kel_index = 0; kel_index < kel_counter; kel_index++)
    {
      const int kel = DistIndice_k[kel_index];
      Uk[kel] += val[kel_index];
      d2Uk[kel] -= hessF00[kel_index];
    }
  }

  inline void computeU3(const ParticleSet& P,
                        int jel,
                        const DistRow& distjI,
                        const DisplRow& displjI,
                        const DistRow& distjk,
                        const DisplRow& displjk,
                        valT& Uj,
                        posT& dUj,
                        valT& d2Uj,
                        Vector<valT>& Uk,
                        gContainer_type& dUk,
                        Vector<valT>& d2Uk,
                        std::vector<int>& ions_nearby,
                        bool triangle = false)
  {
    constexpr valT czero(0);

    Uj   = czero;
    dUj  = posT();
    d2Uj = czero;

    const int jg = P.GroupID[jel];

    const int kelmax = triangle ? jel : Nelec;
    std::fill_n(Uk.data(), kelmax, czero);
    std::fill_n(d2Uk.data(), kelmax, czero);
    for (int idim = 0; idim < OHMMS_DIM; ++idim)
      std::fill_n(dUk.data(idim), kelmax, czero);

    ions_nearby.clear();
    for (int iat = 0; iat < Nion; ++iat)
      if (distjI[iat] < Ion_cutoff[iat])
        ions_nearby.push_back(iat);

    for (int kg = 0; kg < eGroups; ++kg)
    {
      int kel_counter = 0;
      for (int iind = 0; iind < ions_nearby.size(); ++iind)
      {
        const int iat      = ions_nearby[iind];
        const int ig       = Ions.GroupID[iat];
        const valT r_jI    = distjI[iat];
        const posT disp_Ij = displjI[iat];
        for (int kind = 0; kind < elecs_inside(kg, iat).size(); kind++)
        {
          const int kel = elecs_inside(kg, iat)[kind];
          if (kel < kelmax && kel != jel)
          {
            DistkI_Compressed[kel_counter]  = elecs_inside_dist(kg, iat)[kind];
            DistjI_Compressed[kel_counter]  = r_jI;
            Distjk_Compressed[kel_counter]  = distjk[kel];
            Disp_kI_Compressed(kel_counter) = elecs_inside_displ(kg, iat)[kind];
            Disp_jI_Compressed(kel_counter) = disp_Ij;
            Disp_jk_Compressed(kel_counter) = displjk[kel];
            DistIndice_k[kel_counter]       = kel;
            kel_counter++;
            if (kel_counter == Nbuffer)
            {
              const FT& feeI(*F(ig, jg, kg));
              computeU3_engine(P, feeI, kel_counter, Uj, dUj, d2Uj, Uk, dUk, d2Uk);
              kel_counter = 0;
            }
          }
        }
        if ((iind + 1 == ions_nearby.size() || ig != Ions.GroupID[ions_nearby[iind + 1]]) && kel_counter > 0)
        {
          const FT& feeI(*F(ig, jg, kg));
          computeU3_engine(P, feeI, kel_counter, Uj, dUj, d2Uj, Uk, dUk, d2Uk);
          kel_counter = 0;
        }
      }
    }
  }

  inline void registerData(ParticleSet& P, WFBufferType& buf) override
  {
    if (Bytes_in_WFBuffer == 0)
    {
      Bytes_in_WFBuffer = buf.current();
      buf.add(Uat.begin(), Uat.end());
      buf.add(dUat.data(), dUat.end());
      buf.add(d2Uat.begin(), d2Uat.end());
      Bytes_in_WFBuffer = buf.current() - Bytes_in_WFBuffer;
      // free local space
      Uat.free();
      dUat.free();
      d2Uat.free();
    }
    else
    {
      buf.forward(Bytes_in_WFBuffer);
    }
  }

  inline LogValue updateBuffer(ParticleSet& P, WFBufferType& buf, bool fromscratch = false) override
  {
    log_value_ = computeGL(P.G, P.L);
    buf.forward(Bytes_in_WFBuffer);
    return log_value_;
  }

  inline void copyFromBuffer(ParticleSet& P, WFBufferType& buf) override
  {
    Uat.attachReference(buf.lendReference<valT>(Nelec), Nelec);
    dUat.attachReference(Nelec, Nelec_padded, buf.lendReference<valT>(Nelec_padded * OHMMS_DIM));
    d2Uat.attachReference(buf.lendReference<valT>(Nelec), Nelec);
    build_compact_list(P);
  }

  LogValue evaluateGL(const ParticleSet& P,
                      ParticleSet::ParticleGradient& G,
                      ParticleSet::ParticleLaplacian& L,
                      bool fromscratch = false) override
  { return log_value_ = computeGL(G, L); }

  void evaluateDerivatives(ParticleSet& P,
                           const OptVariables& optvars,
                           Vector<ValueType>& dlogpsi,
                           Vector<ValueType>& dhpsioverpsi) override
  {
    resizeWFOptVectors();

    bool recalculate(false);
    for (int k = 0; k < myVars.size(); ++k)
    {
      int kk = myVars.where(k);
      if (kk < 0)
        continue;
      recalculate = true;
    }

    if (recalculate)
    {
      constexpr valT czero(0);
      constexpr valT cone(1);
      constexpr valT cminus(-1);
      constexpr valT ctwo(2);
      constexpr valT lapfac = OHMMS_DIM - cone;

      const auto& ee_table  = P.getDistTableAA(ee_Table_ID_);
      const auto& ee_dists  = ee_table.getDistances();
      const auto& ee_displs = ee_table.getDisplacements();

      build_compact_list(P);

      dLogPsi    = czero;
      gradLogPsi = PosType();
      lapLogPsi  = czero;

      for (int iat = 0; iat < Nion; ++iat)
      {
        const int ig = Ions.GroupID[iat];
        for (int jg = 0; jg < eGroups; ++jg)
          for (int jind = 0; jind < elecs_inside(jg, iat).size(); jind++)
          {
            const int jel       = elecs_inside(jg, iat)[jind];
            const valT r_Ij     = elecs_inside_dist(jg, iat)[jind];
            const posT disp_Ij  = cminus * elecs_inside_displ(jg, iat)[jind];
            const valT r_Ij_inv = cone / r_Ij;

            for (int kg = 0; kg < eGroups; ++kg)
              for (int kind = 0; kind < elecs_inside(kg, iat).size(); kind++)
              {
                const int kel = elecs_inside(kg, iat)[kind];
                if (kel < jel)
                {
                  const valT r_Ik     = elecs_inside_dist(kg, iat)[kind];
                  const posT disp_Ik  = cminus * elecs_inside_displ(kg, iat)[kind];
                  const valT r_Ik_inv = cone / r_Ik;

                  const valT r_jk     = ee_dists[jel][kel];
                  const posT disp_jk  = ee_displs[jel][kel];
                  const valT r_jk_inv = cone / r_jk;

                  FT& func = *F(ig, jg, kg);
                  int idx  = J3UniqueIndex[F(ig, jg, kg)];
                  func.evaluateDerivatives(r_jk, r_Ij, r_Ik, du_dalpha[idx], dgrad_dalpha[idx], dhess_dalpha[idx]);
                  int first                               = VarOffset(ig, jg, kg).first;
                  int last                                = VarOffset(ig, jg, kg).second;
                  std::vector<RealType>& dlog             = du_dalpha[idx];
                  std::vector<PosType>& dgrad             = dgrad_dalpha[idx];
                  std::vector<Tensor<RealType, 3>>& dhess = dhess_dalpha[idx];

                  for (int p = first, ip = 0; p < last; p++, ip++)
                  {
                    RealType& dval          = dlog[ip];
                    PosType& dg             = dgrad[ip];
                    Tensor<RealType, 3>& dh = dhess[ip];

                    dg[0] *= r_jk_inv;
                    dg[1] *= r_Ij_inv;
                    dg[2] *= r_Ik_inv;

                    PosType gr_ee = dg[0] * disp_jk;

                    gradLogPsi(p, jel) -= dg[1] * disp_Ij - gr_ee;
                    lapLogPsi(p, jel) -=
                        (dh(0, 0) + lapfac * dg[0] - ctwo * dh(0, 1) * dot(disp_jk, disp_Ij) * r_jk_inv * r_Ij_inv +
                         dh(1, 1) + lapfac * dg[1]);

                    gradLogPsi(p, kel) -= dg[2] * disp_Ik + gr_ee;
                    lapLogPsi(p, kel) -=
                        (dh(0, 0) + lapfac * dg[0] + ctwo * dh(0, 2) * dot(disp_jk, disp_Ik) * r_jk_inv * r_Ik_inv +
                         dh(2, 2) + lapfac * dg[2]);

                    dLogPsi[p] -= dval;
                  }
                }
              }
          }
      }

      for (int k = 0; k < myVars.size(); ++k)
      {
        int kk = myVars.where(k);
        if (kk < 0)
          continue;
        dlogpsi[kk]  = (ValueType)dLogPsi[k];
        RealType sum = 0.0;
        for (int i = 0; i < Nelec; i++)
        {
#if defined(QMC_COMPLEX)
          sum -= 0.5 * lapLogPsi(k, i);
          for (int jdim = 0; jdim < OHMMS_DIM; ++jdim)
            sum -= P.G[i][jdim].real() * gradLogPsi(k, i)[jdim];
#else
          sum -= 0.5 * lapLogPsi(k, i) + dot(P.G[i], gradLogPsi(k, i));
#endif
        }
        dhpsioverpsi[kk] = (ValueType)sum;
      }
    }
  }

  void evaluateDerivativesWF(ParticleSet& P, const OptVariables& optvars, Vector<ValueType>& dlogpsi) override
  {
    resizeWFOptVectors();

    bool recalculate(false);
    for (int k = 0; k < myVars.size(); ++k)
    {
      int kk = myVars.where(k);
      if (kk < 0)
        continue;
      recalculate = true;
    }

    if (recalculate)
    {
      constexpr valT czero(0);
      constexpr valT cone(1);
      constexpr valT cminus(-1);
      constexpr valT ctwo(2);
      constexpr valT lapfac = OHMMS_DIM - cone;

      const auto& ee_table  = P.getDistTableAA(ee_Table_ID_);
      const auto& ee_dists  = ee_table.getDistances();
      const auto& ee_displs = ee_table.getDisplacements();

      build_compact_list(P);

      dLogPsi    = czero;
      gradLogPsi = PosType();
      lapLogPsi  = czero;

      for (int iat = 0; iat < Nion; ++iat)
      {
        const int ig = Ions.GroupID[iat];
        for (int jg = 0; jg < eGroups; ++jg)
          for (int jind = 0; jind < elecs_inside(jg, iat).size(); jind++)
          {
            const int jel       = elecs_inside(jg, iat)[jind];
            const valT r_Ij     = elecs_inside_dist(jg, iat)[jind];
            const posT disp_Ij  = cminus * elecs_inside_displ(jg, iat)[jind];
            const valT r_Ij_inv = cone / r_Ij;

            for (int kg = 0; kg < eGroups; ++kg)
              for (int kind = 0; kind < elecs_inside(kg, iat).size(); kind++)
              {
                const int kel = elecs_inside(kg, iat)[kind];
                if (kel < jel)
                {
                  const valT r_Ik     = elecs_inside_dist(kg, iat)[kind];
                  const posT disp_Ik  = cminus * elecs_inside_displ(kg, iat)[kind];
                  const valT r_Ik_inv = cone / r_Ik;

                  const valT r_jk     = ee_dists[jel][kel];
                  const posT disp_jk  = ee_displs[jel][kel];
                  const valT r_jk_inv = cone / r_jk;

                  FT& func = *F(ig, jg, kg);
                  int idx  = J3UniqueIndex[F(ig, jg, kg)];
                  func.evaluateDerivatives(r_jk, r_Ij, r_Ik, du_dalpha[idx], dgrad_dalpha[idx], dhess_dalpha[idx]);
                  int first                   = VarOffset(ig, jg, kg).first;
                  int last                    = VarOffset(ig, jg, kg).second;
                  std::vector<RealType>& dlog = du_dalpha[idx];

                  for (int p = first, ip = 0; p < last; p++, ip++)
                  {
                    RealType& dval = dlog[ip];
                    dLogPsi[p] -= dval;
                  }
                }
              }
          }
      }

      for (int k = 0; k < myVars.size(); ++k)
      {
        int kk = myVars.where(k);
        if (kk < 0)
          continue;
        dlogpsi[kk] = (ValueType)dLogPsi[k];
      }
    }
  }

  void evaluateDerivRatios(const VirtualParticleSet& VP,
                           const OptVariables& optvars,
                           std::vector<ValueType>& ratios,
                           Matrix<ValueType>& dratios) override
  {
    assert(VP.getTotalNum() == ratios.size());
    evaluateRatios(VP, ratios);

    bool recalculate(false);
    for (int k = 0; k < myVars.size(); ++k)
    {
      int kk = myVars.where(k);
      if (kk < 0)
        continue;
      recalculate = true;
    }

    if (recalculate)
    {
      constexpr valT czero(0);

      const auto& refPS    = VP.getRefPS();
      const auto& ee_dists = refPS.getDistTableAA(ee_Table_ID_).getDistances();
      const auto& ei_dists = refPS.getDistTableAB(ei_Table_ID_).getDistances();

      const auto& vpe_dists = VP.getDistTableAB(ee_Table_ID_).getDistances();
      const auto& vpi_dists = VP.getDistTableAB(ei_Table_ID_).getDistances();

      const int nVP = VP.getTotalNum();
      std::vector<Vector<RealType>> dLogPsi_vp(nVP);
      for (auto& dLogPsi : dLogPsi_vp)
      {
        dLogPsi.resize(myVars.size());
        dLogPsi = czero;
      }

      const int kel = VP.refPtcl;
      const int kg  = refPS.getGroupID(kel);

      for (int iat = 0; iat < Nion; ++iat)
      {
        const int ig = Ions.getGroupID(iat);
        for (int jg = 0; jg < eGroups; ++jg)
        {
          FT& func             = *F(ig, jg, kg);
          const size_t nparams = func.getNumParameters();
          std::vector<RealType> dlog_ref(nparams), dlog(nparams);
          for (int jind = 0; jind < elecs_inside(jg, iat).size(); jind++)
          {
            const int jel = elecs_inside(jg, iat)[jind];
            if (jel == kel)
              continue;
            const valT r_Ij     = elecs_inside_dist(jg, iat)[jind];
            const valT r_Ik_ref = ei_dists[kel][iat];
            const valT r_jk_ref = jel < kel ? ee_dists[kel][jel] : ee_dists[jel][kel];

            if (!func.evaluateDerivatives(r_jk_ref, r_Ij, r_Ik_ref, dlog_ref))
              std::fill(dlog_ref.begin(), dlog_ref.end(), czero);

            for (int ivp = 0; ivp < nVP; ivp++)
            {
              const valT r_Ik = vpi_dists[ivp][iat];
              const valT r_jk = vpe_dists[ivp][jel];
              if (!func.evaluateDerivatives(r_jk, r_Ij, r_Ik, dlog))
                std::fill(dlog.begin(), dlog.end(), czero);
              const int first = VarOffset(ig, jg, kg).first;
              const int last  = VarOffset(ig, jg, kg).second;
              for (int p = first, ip = 0; p < last; p++, ip++)
                dLogPsi_vp[ivp][p] -= dlog[ip] - dlog_ref[ip];
            }
          }
        }
      }

      for (int k = 0; k < myVars.size(); ++k)
      {
        int kk = myVars.where(k);
        if (kk < 0)
          continue;
        for (int ivp = 0; ivp < nVP; ivp++)
          dratios[ivp][kk] = (ValueType)dLogPsi_vp[ivp][k];
      }
    }
  }

  inline GradType evalGradSource(ParticleSet& P, ParticleSet& source, int isrc) override
  {
    constexpr valT czero(0);
    constexpr valT cone(1);
    constexpr valT cminus(-1);
    constexpr valT ctwo(2);
    constexpr valT lapfac = OHMMS_DIM - cone;

    const auto& ee_table  = P.getDistTableAA(ee_Table_ID_);
    const auto& ee_dists  = ee_table.getDistances();
    const auto& ee_displs = ee_table.getDisplacements();

    TinyVector<RealType, 3> u3grad;
    Tensor<RealType, 3> u3hess;
    const int iat = isrc;

    posT ion_deriv(0);
    const int ig = Ions.GroupID[iat];
    for (int jg = 0; jg < eGroups; ++jg)
      for (int jind = 0; jind < elecs_inside(jg, iat).size(); jind++)
      {
        const int jel       = elecs_inside(jg, iat)[jind];
        const valT r_Ij     = elecs_inside_dist(jg, iat)[jind];
        const posT disp_Ij  = cminus * elecs_inside_displ(jg, iat)[jind];
        const valT r_Ij_inv = cone / r_Ij;

        for (int kg = 0; kg < eGroups; ++kg)
          for (int kind = 0; kind < elecs_inside(kg, iat).size(); kind++)
          {
            const FT& feeI(*F(ig, jg, kg));
            const int kel = elecs_inside(kg, iat)[kind];
            if (kel < jel)
            {
              const valT r_Ik     = elecs_inside_dist(kg, iat)[kind];
              const posT disp_Ik  = cminus * elecs_inside_displ(kg, iat)[kind];
              const valT r_Ik_inv = cone / r_Ik;

              const valT r_jk     = ee_dists[jel][kel];
              const posT disp_jk  = ee_displs[jel][kel];
              const valT r_jk_inv = cone / r_jk;
              feeI.evaluate(r_jk, r_Ij, r_Ik, u3grad, u3hess);
              ion_deriv += u3grad[1] * disp_Ij * r_Ij_inv + u3grad[2] * disp_Ik * r_Ik_inv;
            }
          }
      }
    return ion_deriv;
  }

  inline GradType evalGradSource(ParticleSet& P,
                                 ParticleSet& source,
                                 int isrc,
                                 TinyVector<ParticleSet::ParticleGradient, OHMMS_DIM>& grad_grad,
                                 TinyVector<ParticleSet::ParticleLaplacian, OHMMS_DIM>& lapl_grad) override
  {
    constexpr valT czero(0);
    constexpr valT cone(1);
    constexpr valT cminus(-1);
    constexpr valT ctwo(2);
    constexpr valT lapfac = OHMMS_DIM - cone;

    const auto& ee_table  = P.getDistTableAA(ee_Table_ID_);
    const auto& ee_dists  = ee_table.getDistances();
    const auto& ee_displs = ee_table.getDisplacements();

    ParticleSet::ParticleGradient G;
    ParticleSet::ParticleLaplacian L;
    G.resize(4);
    L.resize(4);

    TinyVector<RealType, 3> grad;
    Tensor<RealType, 3> hess;
    TinyVector<Tensor<RealType, 3>, 3> d3;

    TinyVector<RealType, 3> e1(1, 0, 0);
    TinyVector<RealType, 3> e2(0, 1, 0);
    TinyVector<RealType, 3> e3(0, 0, 1);

    TinyVector<TinyVector<RealType, 3>, 3> identmat(e1, e2, e3);

    const int iat = isrc;

    posT ion_deriv(0);
    const int ig = Ions.GroupID[iat];
    for (int jg = 0; jg < eGroups; ++jg)
      for (int jind = 0; jind < elecs_inside(jg, iat).size(); jind++)
      {
        const int jel           = elecs_inside(jg, iat)[jind];
        const valT r_Ij         = elecs_inside_dist(jg, iat)[jind];
        const posT disp_Ij      = cminus * elecs_inside_displ(jg, iat)[jind];
        const valT r_Ij_inv     = cone / r_Ij;
        const posT disp_Ij_unit = disp_Ij * r_Ij_inv;

        for (int kg = 0; kg < eGroups; ++kg)
          for (int kind = 0; kind < elecs_inside(kg, iat).size(); kind++)
          {
            const FT& feeI(*F(ig, jg, kg));
            const int kel = elecs_inside(kg, iat)[kind];
            if (kel < jel)
            {
              const valT r_Ik         = elecs_inside_dist(kg, iat)[kind];
              const posT disp_Ik      = cminus * elecs_inside_displ(kg, iat)[kind];
              const valT r_Ik_inv     = cone / r_Ik;
              const posT disp_Ik_unit = disp_Ik * r_Ik_inv;

              const valT r_jk         = ee_dists[jel][kel];
              const posT disp_jk      = ee_displs[jel][kel];
              const valT r_jk_inv     = cone / r_jk;
              const posT disp_jk_unit = disp_jk * r_jk_inv;

              const valT dot_ujk_uIj = dot(disp_jk_unit, disp_Ij_unit);
              const valT dot_ujk_uIk = dot(disp_jk_unit, disp_Ik_unit);
              grad                   = 0.0;
              hess                   = 0.0;
              d3                     = 0.0;
              feeI.evaluate(r_jk, r_Ij, r_Ik, grad, hess, d3);
              ion_deriv += grad[1] * disp_Ij * r_Ij_inv + grad[2] * disp_Ik * r_Ik_inv;

              for (int idim = 0; idim < OHMMS_DIM; idim++)
              {
                const posT igrad_r_Ij_unit =
                    -(identmat[idim] * r_Ij_inv - disp_Ij * disp_Ij[idim] * r_Ij_inv * r_Ij_inv * r_Ij_inv);
                const posT igrad_r_Ik_unit =
                    -(identmat[idim] * r_Ik_inv - disp_Ik * disp_Ik[idim] * r_Ik_inv * r_Ik_inv * r_Ik_inv);
                const posT igrad_g0 = -(hess(0, 1) * disp_Ij_unit[idim] + hess(0, 2) * disp_Ik_unit[idim]);
                const posT igrad_g1 = -(hess(1, 1) * disp_Ij_unit[idim] + hess(1, 2) * disp_Ik_unit[idim]);
                const posT igrad_g2 = -(hess(1, 2) * disp_Ij_unit[idim] + hess(2, 2) * disp_Ik_unit[idim]);

                const posT igrad_h00 =
                    -(d3[0](0, 1) * disp_Ij[idim] * r_Ij_inv + d3[0](0, 2) * disp_Ik[idim] * r_Ik_inv);
                const posT igrad_h11 =
                    -(d3[1](1, 1) * disp_Ij[idim] * r_Ij_inv + d3[1](1, 2) * disp_Ik[idim] * r_Ik_inv);
                const posT igrad_h22 =
                    -(d3[1](2, 2) * disp_Ij[idim] * r_Ij_inv + d3[2](2, 2) * disp_Ik[idim] * r_Ik_inv);
                const posT igrad_h01 =
                    -(d3[0](1, 1) * disp_Ij[idim] * r_Ij_inv + d3[0](1, 2) * disp_Ik[idim] * r_Ik_inv);
                const posT igrad_h02 =
                    -(d3[0](1, 2) * disp_Ij[idim] * r_Ij_inv + d3[0](2, 2) * disp_Ik[idim] * r_Ik_inv);
                const posT igrad_dot_ujk_uIj = -(disp_jk_unit[idim] - dot_ujk_uIj * disp_Ij_unit) * r_Ij_inv;
                const posT igrad_dot_ujk_uIk = -(disp_jk_unit[idim] - dot_ujk_uIk * disp_Ik_unit) * r_Ik_inv;

                grad_grad[idim][jel] -= igrad_g1 * disp_Ij_unit + grad[1] * igrad_r_Ij_unit - igrad_g0 * disp_jk_unit;
                grad_grad[idim][kel] -= igrad_g2 * disp_Ik_unit + grad[2] * igrad_r_Ik_unit + igrad_g0 * disp_jk_unit;

                lapl_grad[idim][jel] -= igrad_h00[idim] + lapfac * igrad_g0[idim] * r_jk_inv -
                    ctwo * (igrad_h01[idim] * dot_ujk_uIj + hess(0, 1) * igrad_dot_ujk_uIj[idim]) + igrad_h11[idim] +
                    lapfac * (igrad_g1[idim] * r_Ij_inv - grad[1] * r_Ij_inv * r_Ij_inv * (-disp_Ij_unit[idim]));
                lapl_grad[idim][kel] -= igrad_h00[idim] + lapfac * igrad_g0[idim] * r_jk_inv +
                    ctwo * (igrad_h02[idim] * dot_ujk_uIk + hess(0, 2) * igrad_dot_ujk_uIk[idim]) + igrad_h22[idim] +
                    lapfac * (igrad_g2[idim] * r_Ik_inv - grad[2] * r_Ik_inv * r_Ik_inv * (-disp_Ik_unit[idim]));
              }
            }
          }
      }
    return ion_deriv;
  }
};

} // namespace qmcplusplus
#endif
