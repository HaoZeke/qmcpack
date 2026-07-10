//////////////////////////////////////////////////////////////////////////////////////
// CUDA dense dual-table eeI batch API.
// ratioGrad: resident e-I tables + temps-only transfers, single-stream H2D/kernel/D2H.
// recompute: multi-walker dense rebuild of Uat/dUat/d2Uat.
// The workspace is thread_local and tagged with the owning multi-walker resource +
// a static-table version so distinct Jastrow instances (or updated parameters)
// never reuse another owner's device-resident tables.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_JEEI_DENSE_CUDA_H
#define QMCPLUSPLUS_JEEI_DENSE_CUDA_H

#include <cstddef>

namespace qmcplusplus
{
namespace jeei_cuda
{

class DenseWorkspace
{
public:
  DenseWorkspace();
  ~DenseWorkspace();
  DenseWorkspace(const DenseWorkspace&)            = delete;
  DenseWorkspace& operator=(const DenseWorkspace&) = delete;

  void ensureCapacity(int nw,
                      int Nelec,
                      int Ne_pad,
                      int Nion,
                      int Ni_pad,
                      int nfun,
                      size_t gamma_pool_sz);

  void uploadStatic(int Nelec,
                    int Nion,
                    int nfun,
                    size_t gamma_pool_sz,
                    const int* e_grp,
                    const int* i_grp,
                    const double* ion_cut,
                    const double* gamma_pool,
                    const int* gamma_offset,
                    const int* gamma_len,
                    const double* fun_cut,
                    const int* fun_NeI,
                    const int* fun_Nee,
                    const int* fun_C);

  void uploadFull(int nw,
                  int Nelec,
                  int Ni_pad,
                  const double* ei_full_r,
                  const double* ei_full_dr);

  /** Upload full ee (lower triangle) + e-I tables and the Uat/dUat/d2Uat state
   *  (mw_allUat batch layout) so the device becomes authoritative mid-block. */
  void uploadFullGeometry(int nw,
                          int Nelec,
                          int Ne_pad,
                          int Ni_pad,
                          const double* ee_full_r,
                          const double* ee_full_dr,
                          const double* ei_full_r,
                          const double* ei_full_dr,
                          const double* uat_state);

  /** D2H the resident Uat/dUat/d2Uat batch into the mw_allUat host buffer. */
  void downloadState(int nw, int Ne_pad, double* uat_state);

  /** Upload electron ([nw][Nelec][3]) and ion ([Nion][3]) coordinates; enables
   *  device-side temp-row generation (open boundary conditions). */
  void uploadCoords(int nw, int Nelec, int Nion, const double* epos, const double* ipos);

  bool hasCoords(const void* owner) const { return coords_resident_ && owner_ == owner; }

  /** Generate ee/eI temp rows on device from resident coordinates and the proposed
   *  positions ([nw][3]); replaces the host temp pack + H2D. */
  void launchTempRows(const void* owner, int jel, int nw, int Nelec, int Ne_pad, int Nion, int Ni_pad,
                      const double* prop);

  /** Device-side accept: old-side dense pass from resident tables, apply new-old
   *  deltas into resident Uat state, patch moved ee/ei rows from resident temps. */
  void launchAccept(const void* owner,
                    int jel,
                    int nw,
                    int Nelec,
                    int Ne_pad,
                    int Nion,
                    int Ni_pad,
                    int eGroups,
                    int nfun,
                    const int* accepted,
                    bool commit_coords);

  /** D2H dUat[jel] for all walkers (3*nw doubles, dim-major per walker). */
  void gatherGrad(const void* owner, int jel, int nw, int Ne_pad, double* grad3);

  /** Patch one electron's e-I row after accept (avoids full re-upload).
   *  No-op unless this workspace's resident tables belong to @p owner. */
  void updateEiRow(const void* owner,
                   int iw,
                   int jel,
                   int Nelec,
                   int Ni_pad,
                   int Nion,
                   const double* r_row,
                   const double* dr_row_3xNi);

  void invalidateFull() { full_resident_ = false; }

  bool hasStatic() const { return static_uploaded_; }
  bool hasFull() const { return full_resident_; }
  /** Resident full e-I tables are valid for this owner (same resource, still uploaded). */
  bool ownsFull(const void* owner) const { return full_resident_ && owner_ == owner; }
  /** Static tables (gamma/cutoffs/species) match this owner at this pack version. */
  bool matchesStatic(const void* owner, unsigned long long ver) const
  {
    return static_uploaded_ && owner_ == owner && static_ver_ == ver;
  }
  void setOwner(const void* owner, unsigned long long ver)
  {
    owner_      = owner;
    static_ver_ = ver;
  }
  bool sameOwner(const void* owner) const { return owner_ == owner; }

  void launchRatioGrad(int jel,
                       int nw,
                       int Nelec,
                       int Ne_pad,
                       int Nion,
                       int Ni_pad,
                       int eGroups,
                       int nfun,
                       const double* ee_temp,
                       const double* ei_temp,
                       const double* ei_full_r,
                       const double* ei_full_dr,
                       bool upload_full,
                       bool copy_uk_host,
                       bool temps_on_device,
                       double* vgl,
                       double* Uk,
                       double* dUk,
                       double* d2Uk);

  /** Full U rebuild: needs full ee lower-triangle + full eI tables packed on host. */
  void launchRecompute(int nw,
                       int Nelec,
                       int Ne_pad,
                       int Nion,
                       int Ni_pad,
                       int eGroups,
                       int nfun,
                       const double* ee_full_r,
                       const double* ee_full_dr,
                       const double* ei_full_r,
                       const double* ei_full_dr,
                       double* Uat,
                       double* dUat,
                       double* d2Uat);

private:
  void freeAll();
  void ensureStream();

  int cap_nw_       = 0;
  int cap_Ne_       = 0;
  int cap_Nep_      = 0;
  int cap_Ni_       = 0;
  int cap_Nip_      = 0;
  int cap_nfun_     = 0;
  size_t cap_gamma_ = 0;

  double *d_ee_ = nullptr, *d_ei_ = nullptr, *d_fr_ = nullptr, *d_fd_ = nullptr;
  double *d_ee_full_r_ = nullptr, *d_ee_full_dr_ = nullptr;
  double *d_ion_cut_ = nullptr, *d_gamma_ = nullptr, *d_fun_cut_ = nullptr;
  double *d_vgl_ = nullptr, *d_Uk_ = nullptr, *d_dUk_ = nullptr, *d_d2Uk_ = nullptr;
  double *d_Uat_ = nullptr, *d_dUat_ = nullptr, *d_d2Uat_ = nullptr;
  double* d_grad_ = nullptr;
  double *d_epos_ = nullptr, *d_ipos_ = nullptr, *d_prop_ = nullptr;
  int* d_accept_  = nullptr;
  int *d_egrp_ = nullptr, *d_igrp_ = nullptr, *d_goff_ = nullptr, *d_glen_ = nullptr;
  int *d_NeI_ = nullptr, *d_Nee_ = nullptr, *d_C_ = nullptr;

  // Single stream per workspace (thread_local): pipeline H2D -> kernel -> D2H, one sync.
  void* stream_ = nullptr; // cudaStream_t without requiring cuda.h in this header

  // Identity of the multi-walker resource whose tables are resident, plus the
  // static-pack version uploaded for it. Guards against silent reuse across
  // Jastrow instances, crowds, or parameter updates sharing this thread.
  const void* owner_             = nullptr;
  unsigned long long static_ver_ = 0;

  bool static_uploaded_ = false;
  bool full_resident_   = false;
  bool coords_resident_ = false;
};

DenseWorkspace& default_workspace();

/** ensureCapacity + static upload on the calling thread's workspace (for callers
 *  that upload geometry/state explicitly before launching kernels). */
DenseWorkspace& prepare_dense_workspace(const void* owner,
                                        unsigned long long static_ver,
                                        int nw,
                                        int Nelec,
                                        int Ne_pad,
                                        int Nion,
                                        int Ni_pad,
                                        int nfun,
                                        const int* e_grp,
                                        const int* i_grp,
                                        const double* ion_cut,
                                        const double* gamma_pool,
                                        const int* gamma_offset,
                                        const int* gamma_len,
                                        const double* fun_cut,
                                        const int* fun_NeI,
                                        const int* fun_Nee,
                                        const int* fun_C);

void launch_dense_ratio_grad(const void* owner,
                             unsigned long long static_ver,
                             int jel,
                             int nw,
                             int Nelec,
                             int Ne_pad,
                             int Nion,
                             int Ni_pad,
                             int eGroups,
                             int iGroups,
                             int nfun,
                             const double* ee_temp,
                             const double* ei_temp,
                             const double* ei_full_r,
                             const double* ei_full_dr,
                             const int* e_grp,
                             const int* i_grp,
                             const double* ion_cut,
                             const double* gamma_pool,
                             const int* gamma_offset,
                             const int* gamma_len,
                             const double* fun_cut,
                             const int* fun_NeI,
                             const int* fun_Nee,
                             const int* fun_C,
                             bool force_full_upload,
                             bool copy_uk_host,
                             bool temps_on_device,
                             double* vgl,
                             double* Uk,
                             double* dUk,
                             double* d2Uk);

void launch_dense_recompute(const void* owner,
                            unsigned long long static_ver,
                            int nw,
                            int Nelec,
                            int Ne_pad,
                            int Nion,
                            int Ni_pad,
                            int eGroups,
                            int iGroups,
                            int nfun,
                            const double* ee_full_r,
                            const double* ee_full_dr,
                            const double* ei_full_r,
                            const double* ei_full_dr,
                            const int* e_grp,
                            const int* i_grp,
                            const double* ion_cut,
                            const double* gamma_pool,
                            const int* gamma_offset,
                            const int* gamma_len,
                            const double* fun_cut,
                            const int* fun_NeI,
                            const int* fun_Nee,
                            const int* fun_C,
                            double* Uat,
                            double* dUat,
                            double* d2Uat);

} // namespace jeei_cuda
} // namespace qmcplusplus
#endif
