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
/** Counts the work computeU does, so the gather can be told apart from the polynomial.
 *
 * The 48.64 s that JeeIOrbitalSoA:J3::NLratio costs on CO2/Cu(110) is one exclusive
 * timer leaf covering both the data-dependent gather over elecs_inside and the
 * polynomial evaluation it feeds. perf cannot separate them because both are inline in
 * this header and merge into the enclosing symbol. Counting triplets gives the
 * denominator a standalone benchmark of the functor needs to be comparable.
 */
struct JeeIWorkTally
{
  static JeeIWorkTally& get()
  {
    static JeeIWorkTally singleton;
    return singleton;
  }

  static bool enabled()
  {
    static const bool on = [] {
      const char* c = std::getenv("QMCPACK_TALLY_J3_WORK");
      return c && *c == '1';
    }();
    return on;
  }

  void add(size_t triplets, size_t nearby)
  {
    calls_.fetch_add(1, std::memory_order_relaxed);
    triplets_.fetch_add(triplets, std::memory_order_relaxed);
    nearby_.fetch_add(nearby, std::memory_order_relaxed);
  }

  ~JeeIWorkTally()
  {
    if (calls_.load() == 0)
      return;
    std::cerr << "J3TALLY computeU calls=" << calls_.load() << " triplets=" << triplets_.load()
              << " nearby ion visits=" << nearby_.load() << std::endl;
  }

private:
  JeeIWorkTally() = default;
  std::atomic<size_t> calls_{0}, triplets_{0}, nearby_{0};
};

/** Device-side mirrors of the state JeeIOrbitalSoA::computeU walks on the host.
 *
 * elecs_inside is Array<std::vector<int>,2>: ragged, and a target region cannot follow
 * it. Everything here is the same information in offsets-plus-values form, refreshed
 * per call rather than kept in sync with accepted moves, which keeps the invariant
 * trivial at the price of one repack per ratio evaluation.
 */
/** Which path mw_evaluateRatios actually took.
 *
 * The device and host paths agree on energies when the device path works AND when it
 * silently falls back, so identical energies alone cannot tell them apart. This says
 * which one ran.
 */
struct JeeIPathTally
{
  static JeeIPathTally& get()
  {
    static JeeIPathTally singleton;
    return singleton;
  }
  static bool enabled()
  {
    static const bool on = [] {
      const char* c = std::getenv("QMCPACK_TALLY_J3_PATH");
      return c && *c == '1';
    }();
    return on;
  }
  void device() { dev_.fetch_add(1, std::memory_order_relaxed); }
  void fallback() { host_.fetch_add(1, std::memory_order_relaxed); }
  ~JeeIPathTally()
  {
    if (dev_.load() || host_.load())
      std::cerr << "J3PATH device=" << dev_.load() << " fallback=" << host_.load() << std::endl;
  }

private:
  JeeIPathTally() = default;
  std::atomic<size_t> dev_{0}, host_{0};
};

template<typename VALT>
struct JeeIMultiWalkerMem : public Resource
{
  Vector<size_t, OffloadPinnedAllocator<size_t>> memb_offsets;
  Vector<int, OffloadPinnedAllocator<int>> memb_elec;
  Vector<VALT, OffloadPinnedAllocator<VALT>> memb_dist;
  Vector<VALT, OffloadPinnedAllocator<VALT>> gamma_flat;
  Vector<size_t, OffloadPinnedAllocator<size_t>> gamma_offset;
  Vector<char, OffloadPinnedAllocator<char>> fn_have;
  Vector<int, OffloadPinnedAllocator<int>> N_eI, N_ee, C;
  Vector<VALT, OffloadPinnedAllocator<VALT>> L;
  Vector<VALT, OffloadPinnedAllocator<VALT>> ion_cutoff;
  Vector<int, OffloadPinnedAllocator<int>> ion_group;
  Vector<int, OffloadPinnedAllocator<int>> vp_walker, vp_jg;
  Vector<VALT, OffloadPinnedAllocator<VALT>> vals;

  size_t memb_walker_stride = 0;

  JeeIMultiWalkerMem() : Resource("JeeIMultiWalkerMem") {}
  JeeIMultiWalkerMem(const JeeIMultiWalkerMem&) : JeeIMultiWalkerMem() {}
  std::unique_ptr<Resource> makeClone() const override { return std::make_unique<JeeIMultiWalkerMem>(*this); }

  /// flatten every walker's elecs_inside into one offsets/values pair
  template<typename WFCPTRS>
  void packMembership(const WFCPTRS& wfcs, int eGroups, int Nion)
  {
    const size_t nw    = wfcs.size();
    memb_walker_stride = static_cast<size_t>(eGroups) * Nion;
    memb_offsets.resize(nw * memb_walker_stride + 1);

    size_t total = 0;
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

    size_t at = 0;
    for (size_t iw = 0; iw < nw; iw++)
    {
      const auto& wfc = *wfcs[iw];
      for (int kg = 0; kg < eGroups; kg++)
        for (int iat = 0; iat < Nion; iat++)
        {
          const auto& els = wfc.getElecsInside(kg, iat);
          const auto& dst = wfc.getElecsInsideDist(kg, iat);
          for (size_t n = 0; n < els.size(); n++, at++)
          {
            memb_elec[at] = els[n];
            memb_dist[at] = dst[n];
          }
        }
    }
    memb_offsets.updateTo();
    memb_elec.updateTo();
    memb_dist.updateTo();
  }

  /// one flat gamma block per (ion group, j group, k group), plus a present/absent flag
  template<typename FARRAY>
  void packFunctors(const FARRAY& F, int eGroups, int iGroups)
  {
    const size_t ncombo = static_cast<size_t>(iGroups) * eGroups * eGroups;
    fn_have.resize(ncombo);
    gamma_offset.resize(ncombo);
    N_eI.resize(ncombo);
    N_ee.resize(ncombo);
    C.resize(ncombo);
    L.resize(ncombo);
    std::fill(fn_have.begin(), fn_have.end(), char(0));
    std::fill(gamma_offset.begin(), gamma_offset.end(), size_t(0));
    std::fill(N_eI.begin(), N_eI.end(), 0);
    std::fill(N_ee.begin(), N_ee.end(), 0);
    std::fill(C.begin(), C.end(), 0);
    std::fill(L.begin(), L.end(), VALT(0));

    size_t gamma_total = 0;
    for (int ig = 0; ig < iGroups; ig++)
      for (int jg = 0; jg < eGroups; jg++)
        for (int kg = 0; kg < eGroups; kg++)
          if (const auto* functor = F(ig, jg, kg))
          {
            const size_t fidx  = (static_cast<size_t>(ig) * eGroups + jg) * eGroups + kg;
            fn_have[fidx]      = char(1);
            gamma_offset[fidx] = gamma_total;
            N_eI[fidx]         = functor->getNeI();
            N_ee[fidx]         = functor->getNee();
            C[fidx]            = functor->getC();
            L[fidx]            = VALT(0.5) * functor->cutoff_radius;
            gamma_total += functor->gammaFlatSize();
          }

    gamma_flat.resize(std::max(size_t(1), gamma_total));
    std::fill(gamma_flat.begin(), gamma_flat.end(), VALT(0));
    for (int ig = 0; ig < iGroups; ig++)
      for (int jg = 0; jg < eGroups; jg++)
        for (int kg = 0; kg < eGroups; kg++)
          if (F(ig, jg, kg))
          {
            const size_t fidx = (static_cast<size_t>(ig) * eGroups + jg) * eGroups + kg;
            F(ig, jg, kg)->copyGammaFlat(gamma_flat.data() + gamma_offset[fidx]);
          }
    gamma_flat.updateTo();
    gamma_offset.updateTo();
    fn_have.updateTo();
    N_eI.updateTo();
    N_ee.updateTo();
    C.updateTo();
    L.updateTo();
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

  JeeIOrbitalSoA(const std::string& obj_name, const ParticleSet& ions, ParticleSet& elecs)
      : WaveFunctionComponent(obj_name),
        ee_Table_ID_(elecs.addTable(elecs, DTModes::NEED_TEMP_DATA_ON_HOST | DTModes::NEED_VP_FULL_TABLE_ON_HOST)),
        ei_Table_ID_(elecs.addTable(ions, DTModes::NEED_FULL_TABLE_ANYTIME | DTModes::NEED_VP_FULL_TABLE_ON_HOST)),
        Ions(ions)
  {
    if (my_name_.empty())
      throw std::runtime_error("JeeIOrbitalSoA object name cannot be empty!");
    // the batched ratio path reads the virtual-particle tables through
    // getMultiWalkerDataPtr, which only the offload tables provide
    const char* off = std::getenv("QMCPACK_DISABLE_J3_OFFLOAD");
    use_offload_ = elecs.getCoordinates().getKind() == DynamicCoordinateKind::DC_POS_OFFLOAD && !(off && *off == '1');
    init(elecs);
  }

  std::string getClassName() const override { return "JeeIOrbitalSoA"; }

  void createResource(ResourceCollection& collection) const override
  { collection.addResource(std::make_unique<JeeIMultiWalkerMem<valT>>()); }

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
    auto eeIcopy = std::make_unique<JeeIOrbitalSoA<FT>>(my_name_, Ions, elecs);
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
   * computeU on the host. On CO2/Cu(110) that measured 207.56 s of thread-summed time
   * for 13,975,174,235 triplets, 14.85 ns each with the gather included, and the
   * polynomial's own 64-iteration dependency chain accounts for nearly all of it. The
   * triplets are mutually independent, which is the case a device is for.
   *
   * The gather runs on the device too. Handing the host the triplets instead would move
   * 14e9 of them, and at three doubles apiece that is hundreds of gigabytes; the filter
   * is cheap in comparison, only 1.05 of 33 ions surviving the cutoff per call.
   */
  void mw_evaluateRatios(const RefVectorWithLeader<WaveFunctionComponent>& wfc_list,
                         const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
                         std::vector<std::vector<ValueType>>& ratios) const override
  {
    if (wfc_list.size() == 0)
      return;
    if (!use_offload_)
    {
      if (JeeIPathTally::enabled())
        JeeIPathTally::get().fallback();
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
      mw_ei = dt_ei.getMultiWalkerDataPtr();
      mw_ee = dt_ee.getMultiWalkerDataPtr();
    }
    catch (...)
    {
      if (JeeIPathTally::enabled())
        JeeIPathTally::get().fallback();
      WaveFunctionComponent::mw_evaluateRatios(wfc_list, vp_list, ratios);
      return;
    }
    if (mw_ei == nullptr || mw_ee == nullptr)
    {
      if (JeeIPathTally::enabled())
        JeeIPathTally::get().fallback();
      WaveFunctionComponent::mw_evaluateRatios(wfc_list, vp_list, ratios);
      return;
    }
    if (JeeIPathTally::enabled())
      JeeIPathTally::get().device();

    const size_t stride_ei = dt_ei.getPerTargetPctlStrideSize();
    const size_t stride_ee = dt_ee.getPerTargetPctlStrideSize();

    // elecs_inside is Array<std::vector<int>,2>, ragged and host only. Flatten it once
    // per call into offsets plus values, which is also what lets the knots share one
    // walk of it: the host path rebuilds this structure for every knot.
    std::vector<const JeeIOrbitalSoA<FT>*> wfcs(nw);
    for (int iw = 0; iw < nw; iw++)
      wfcs[iw] = &wfc_list.getCastedElement<JeeIOrbitalSoA<FT>>(iw);
    mem.packMembership(wfcs, wfc_leader.eGroups, wfc_leader.Nion);
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
    auto* gamma_off  = mem.gamma_offset.data();
    auto* fn_have    = mem.fn_have.data();
    auto* fn_N_eI    = mem.N_eI.data();
    auto* fn_N_ee    = mem.N_ee.data();
    auto* fn_C       = mem.C.data();
    auto* fn_L       = mem.L.data();
    auto* ion_cut    = mem.ion_cutoff.data();
    auto* ion_grp    = mem.ion_group.data();
    auto* vals       = mem.vals.data();
    auto* walker_of  = mem.vp_walker.data();
    auto* jg_of      = mem.vp_jg.data();
    auto* refp       = mw_refPctls.data();

    const size_t memb_stride = mem.memb_walker_stride;
    const size_t nfun        = static_cast<size_t>(iGroups) * eGroups * eGroups;
    const size_t gamma_size  = mem.gamma_flat.size();
    const size_t n_memb      = mem.memb_elec.size();
    const size_t n_off       = mem.memb_offsets.size();

    PRAGMA_OFFLOAD("omp target teams distribute \
                    map(to: refp[:nVPs], walker_of[:nVPs], jg_of[:nVPs]) \
                    map(to: memb_off[:n_off], memb_elec[:n_memb], memb_dist[:n_memb]) \
                    map(to: gamma_flat[:gamma_size], gamma_off[:nfun], fn_have[:nfun], \
                            fn_N_eI[:nfun], fn_N_ee[:nfun], fn_C[:nfun], fn_L[:nfun]) \
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
          const RealType* grow = gamma_flat + gamma_off[fidx];
          const size_t slot    = memb_base + static_cast<size_t>(kg) * Nion + iat;
          const size_t begin   = memb_off[slot];
          const size_t end     = memb_off[slot + 1];
          for (size_t idx = begin; idx < end; idx++)
          {
            const int kel = memb_elec[idx];
            if (kel == jel)
              continue;
            sum += FT::evaluateV_impl(ee_row[kel], r_jI, memb_dist[idx], grow, fn_N_eI[fidx], fn_N_ee[fidx], fn_C[fidx],
                                      fn_L[fidx]);
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
  const std::vector<valT>& getElecsInsideDist(int kg, int iat) const { return elecs_inside_dist(kg, iat); }

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

    const int ig = P.GroupID[iat];
    // update compact list elecs_inside
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

    size_t tally_triplets = 0;
    valT Uj               = valT(0);
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
              tally_triplets += kel_counter;
              Uj += feeI.evaluateV(kel_counter, Distjk_Compressed.data(), DistjI_Compressed.data(),
                                   DistkI_Compressed.data());
              kel_counter = 0;
            }
          }
        }
        if ((iind + 1 == ions_nearby.size() || ig != Ions.GroupID[ions_nearby[iind + 1]]) && kel_counter > 0)
        {
          const FT& feeI(*F(ig, jg, kg));
          tally_triplets += kel_counter;
          Uj +=
              feeI.evaluateV(kel_counter, Distjk_Compressed.data(), DistjI_Compressed.data(), DistkI_Compressed.data());
          kel_counter = 0;
        }
      }
    }
    if (JeeIWorkTally::enabled())
      JeeIWorkTally::get().add(tally_triplets, ions_nearby.size());
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
