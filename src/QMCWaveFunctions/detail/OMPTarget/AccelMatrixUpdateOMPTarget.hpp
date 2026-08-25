//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2024 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_COMPUTE_MATRIX_UPDATE_OMPTARGET_H
#define QMCPLUSPLUS_COMPUTE_MATRIX_UPDATE_OMPTARGET_H

#include <algorithm>
#include <QueueAliases.hpp>
#include "type_traits/template_types.hpp"

namespace qmcplusplus
{

namespace compute
{

template<typename T>
void copyAinvRow_saveGL_batched(Queue<PlatformKind::OMPTARGET>& queue,
                                const int rowchanged,
                                const int n,
                                const T* const Ainv[],
                                const int lda,
                                T* const temp[],
                                T* const rcopy[],
                                const T* const phi_vgl_in[],
                                const size_t phi_vgl_stride,
                                T* const dphi_out[],
                                T* const d2phi_out[],
                                const int batch_count,
                                const char* accept_mask = nullptr)
{
  /* One combined construct rather than teams distribute wrapped around a separate parallel
   * for. The split form leaves a sequential region between them, and a target region shaped
   * that way compiles to generic mode: one thread runs the sequential part behind a state
   * machine while the rest of the block waits, which is what the runtime reports for this
   * kernel and what holds it to a handful of active warps per SM.
   *
   * The scalar update the sequential region carried is done by one iteration of the
   * collapsed loop. It is safe there because no iteration reads temp: the loop reads Ainv
   * and phi_vgl_in and writes rcopy, dphi_out and d2phi_out.
   */
  PRAGMA_OFFLOAD("omp target teams distribute parallel for collapse(2) \
                  is_device_ptr(Ainv, temp, rcopy, phi_vgl_in, dphi_out, d2phi_out, accept_mask)")
  for (size_t iw = 0; iw < batch_count; iw++)
    for (size_t col_id = 0; col_id < n; col_id++)
    {
      /* With a mask the caller runs over every walker rather than packing the accepted ones
       * first, which is what lets the grid and the pointer offsets stop depending on how many
       * accepted. A rejected walker has no output rows, so it is skipped before anything is
       * dereferenced.
       */
      if (accept_mask && !accept_mask[iw])
        continue;
      const T* __restrict__ Ainv_iw   = Ainv[iw];
      T* __restrict__ rcopy_iw        = rcopy[iw];
      const T* __restrict__ phi_in_iw = phi_vgl_in[iw];
      T* __restrict__ dphi_out_iw     = dphi_out[iw];
      T* __restrict__ d2phi_out_iw    = d2phi_out[iw];

      if (col_id == 0)
      {
        T* __restrict__ temp_iw = temp[iw];
        temp_iw[rowchanged]     = temp_iw[rowchanged] - T(1);
      }

      rcopy_iw[col_id] = Ainv_iw[rowchanged * lda + col_id];

      // the following copying data on the device is not part of SM-1
      // it is intended to copy dphiV and d2phiV from temporary to final without a separate kernel.
      dphi_out_iw[col_id * 3]     = phi_in_iw[col_id + phi_vgl_stride];
      dphi_out_iw[col_id * 3 + 1] = phi_in_iw[col_id + phi_vgl_stride * 2];
      dphi_out_iw[col_id * 3 + 2] = phi_in_iw[col_id + phi_vgl_stride * 3];
      d2phi_out_iw[col_id]        = phi_in_iw[col_id + phi_vgl_stride * 4];
    }
}

template<typename T>
void calcGradients_batched(Queue<PlatformKind::OMPTARGET>& queue,
                           const int n,
                           const T* const Ainvrow[],
                           const T* const dpsiMrow[],
                           T* const grads_now,
                           const int batch_count)
{
  /* One thread per walker and dimension, each summing the whole row, rather than a team
   * reducing over the row with the sums written outside the parallel region.
   *
   * The second shape puts a sequential region on both sides of an inner parallel for, and
   * the compiler answers that with generic mode: one thread walks the sequential parts
   * behind a state machine while the rest of the block waits. The runtime reports the mode
   * per launch, and this kernel was generic. Three numbers per walker over n orbitals is a
   * few thousand multiplies in total, so the arithmetic was never what this cost.
   */
  PRAGMA_OFFLOAD("omp target teams distribute parallel for collapse(2) \
                  is_device_ptr(Ainvrow, dpsiMrow, grads_now)")
  for (size_t iw = 0; iw < batch_count; iw++)
    for (int idim = 0; idim < 3; idim++)
    {
      const T* __restrict__ invRow    = Ainvrow[iw];
      const T* __restrict__ dpsiM_row = dpsiMrow[iw];

      T sum = 0;
      for (size_t col_id = 0; col_id < n; col_id++)
        sum += invRow[col_id] * dpsiM_row[col_id * 3 + idim];

      grads_now[iw * 3 + idim] = sum;
    }
}

template<typename T>
void add_delay_list_save_sigma_VGL_batched(Queue<PlatformKind::OMPTARGET>& queue,
                                           int* const delay_list[],
                                           const int rowchanged,
                                           const int delay_count,
                                           T* const binv[],
                                           const int binv_lda,
                                           const T* const ratio_inv,
                                           const T* const phi_vgl_in[],
                                           const size_t phi_vgl_stride,
                                           T* const phi_out[],
                                           T* const dphi_out[],
                                           T* const d2phi_out[],
                                           const int norb,
                                           const int n_accepted,
                                           const int batch_count,
                                           const char* const accept_mask = nullptr)
{
  // With a mask the walkers stay in their natural order and the branch is taken per walker
  // on the device, so the caller does not have to partition them by a count it can only
  // know on the host. Without one the packed order is used, accepted first, as before.
  PRAGMA_OFFLOAD("omp target teams distribute \
                  is_device_ptr(delay_list, binv, ratio_inv, phi_vgl_in, phi_out, dphi_out, d2phi_out, accept_mask)")
  for (size_t iw = 0; iw < batch_count; iw++)
    if (accept_mask ? accept_mask[iw] != 0 : iw < n_accepted)
    {
      // real accept, settle y and Z
      int* __restrict__ delay_list_iw = delay_list[iw];
      T* __restrict__ binvrow_iw      = binv[iw] + delay_count * binv_lda;
      const T* __restrict__ phi_in_iw = phi_vgl_in[iw];
      T* __restrict__ phi_out_iw      = phi_out[iw];
      T* __restrict__ dphi_out_iw     = dphi_out[iw];
      T* __restrict__ d2phi_out_iw    = d2phi_out[iw];

      delay_list_iw[delay_count] = rowchanged;
      binvrow_iw[delay_count]    = ratio_inv[iw];

      PRAGMA_OFFLOAD("omp parallel for")
      for (size_t col_id = 0; col_id < delay_count; col_id++)
        binvrow_iw[col_id] *= ratio_inv[iw];

      PRAGMA_OFFLOAD("omp parallel for")
      for (size_t col_id = 0; col_id < norb; col_id++)
      {
        // copy phiV, dphiV and d2phiV from temporary to final without a separate kernel.
        phi_out_iw[col_id]          = phi_in_iw[col_id];
        dphi_out_iw[col_id * 3]     = phi_in_iw[col_id + phi_vgl_stride];
        dphi_out_iw[col_id * 3 + 1] = phi_in_iw[col_id + phi_vgl_stride * 2];
        dphi_out_iw[col_id * 3 + 2] = phi_in_iw[col_id + phi_vgl_stride * 3];
        d2phi_out_iw[col_id]        = phi_in_iw[col_id + phi_vgl_stride * 4];
      }
    }
    else
    {
      // fake accept. Set Y, Z with zero and x with 1
      T* __restrict__ binv_iw = binv[iw];
      PRAGMA_OFFLOAD("omp parallel for")
      for (size_t col_id = 0; col_id < delay_count; col_id++)
        binv_iw[delay_count * binv_lda + col_id] = binv_iw[delay_count + binv_lda * col_id] = T(0);

      int* __restrict__ delay_list_iw               = delay_list[iw];
      binv_iw[delay_count * binv_lda + delay_count] = T(1);
      delay_list_iw[delay_count]                    = -1;

      T* __restrict__ Urow_iw = phi_out[iw];
      PRAGMA_OFFLOAD("omp parallel for")
      for (size_t col_id = 0; col_id < norb; col_id++)
      {
        Urow_iw[col_id] = T(0);
      }
    }
}


template<typename T>
void applyW_batched(Queue<PlatformKind::OMPTARGET>& queue,
                    const int* const delay_list[],
                    const int delay_count,
                    T* const tempMat[],
                    const int lda,
                    const int batch_count)
{
  PRAGMA_OFFLOAD("omp target teams distribute is_device_ptr(delay_list, tempMat)")
  for (size_t iw = 0; iw < batch_count; iw++)
  {
    const int* __restrict__ delay_list_iw = delay_list[iw];
    T* __restrict__ tempMat_iw            = tempMat[iw];

    PRAGMA_OFFLOAD("omp parallel for")
    for (size_t col_id = 0; col_id < delay_count; col_id++)
    {
      const int row_id = delay_list_iw[col_id];
      if (row_id >= 0)
        tempMat_iw[row_id * lda + col_id] = tempMat_iw[row_id * lda + col_id] - T(1);
    }
  }
}


/// walkers a crowd must hold before packing the spans repays its kernel launch
inline constexpr int gather_min_batch = 2;

/** span size above which packing costs more than the round trips it merges
 *
 * Round trips saved grow with the walkers in the crowd, while the copy out of the
 * staging buffer grows with walkers times span, so a large enough span turns the trade
 * around whatever the crowd size. A standalone comparison of the two paths on this
 * device places the crossover between 256 and 384 KiB per span, at 8, 16 and 64 walkers
 * alike, which is what the growth rates predict: the crowd size cancels.
 */
inline constexpr size_t gather_max_span_bytes = 256 * 1024;

/** copy the same span of every walker's container to the host in one transfer
 *
 * @param items    one dual space container per walker
 * @param n        elements to copy from each
 * @param offset   where the span starts in each
 * @param ptrs     crowd scratch, device addresses of the spans
 * @param staging  crowd scratch, the spans packed back to back
 *
 * An update carries the cost of a round trip rather than the cost of its bytes, so a
 * span of a few hundred bytes per walker costs the crowd one round trip each. Packing
 * the spans on the device leaves one transfer to carry all of them.
 *
 * Packing only pays once there are round trips to merge. It adds a kernel launch, an
 * upload of the address list and a host side copy out of the staging buffer, all of
 * which the direct path avoids, so a crowd holding one walker transfers the same bytes
 * and pays the extra. Below @ref gather_min_batch walkers, and above
 * @ref gather_max_span_bytes per span, the spans go directly.
 */
template<class CONTAINER, class PTRVEC, class STAGEVEC>
void copyEachToHost(Queue<PlatformKind::OMPTARGET>& queue,
                    const RefVector<CONTAINER>& items,
                    const size_t n,
                    const size_t offset,
                    PTRVEC& ptrs,
                    STAGEVEC& staging)
{
  const int batch_count = items.size();
  if (batch_count == 0 || n == 0)
    return;

  if (batch_count < gather_min_batch || n * sizeof(typename CONTAINER::value_type) > gather_max_span_bytes)
  {
    for (int iw = 0; iw < batch_count; iw++)
      queue.enqueueD2H(items[iw].get(), n, offset);
    queue.sync();
    return;
  }

  /* The spans keep their device addresses between calls, so the list already on the
   * device usually still describes the crowd. Comparing it walks a few pointers on the
   * host and saves a round trip whenever it holds. A resize leaves the contents
   * unspecified, so a crowd that changed size always uploads.
   */
  const bool same_size = (ptrs.size() == static_cast<size_t>(batch_count));
  ptrs.resize(batch_count);
  staging.resize(n * batch_count);
  bool addresses_current = same_size;
  for (int iw = 0; iw < batch_count; iw++)
  {
    auto* const span  = items[iw].get().device_data() + offset;
    addresses_current = addresses_current && (ptrs[iw] == span);
    ptrs[iw]          = span;
  }
  if (!addresses_current)
    ptrs.updateTo();

  auto* src_list = ptrs.device_data();
  auto* packed   = staging.device_data();
  PRAGMA_OFFLOAD("omp target teams distribute is_device_ptr(src_list, packed)")
  for (int iw = 0; iw < batch_count; iw++)
  {
    const auto* __restrict__ src = src_list[iw];
    auto* __restrict__ dest      = packed + static_cast<size_t>(iw) * n;

    PRAGMA_OFFLOAD("omp parallel for")
    for (size_t i = 0; i < n; i++)
      dest[i] = src[i];
  }

  queue.enqueueD2H(staging);
  queue.sync();

  for (int iw = 0; iw < batch_count; iw++)
    std::copy_n(staging.data() + static_cast<size_t>(iw) * n, n, items[iw].get().data() + offset);
}

} // namespace compute
} // namespace qmcplusplus
#endif
