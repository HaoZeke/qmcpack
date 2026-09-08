//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_DETERMINEDEFAULTDEVICENUM_H
#define QMCPLUSPLUS_DETERMINEDEFAULTDEVICENUM_H

#include <string>

namespace qmcplusplus
{
/** message for two device managers disagreeing about how many devices exist
 *
 * The managers run in sequence and share one count, so the second to run
 * compares what it sees against the first one's record. Reporting only that
 * they disagree leaves the reader without either number, and the two cases
 * behind it want different actions: a mismatch of two nonzero counts means the
 * runtimes see different sets of devices, while a count of zero against a
 * nonzero record means this runtime found no usable device at all. For an
 * offload runtime that is what a binary built for another architecture looks
 * like, since the image simply does not match and no device is reported.
 */
inline std::string deviceCountMismatchMessage(const char* runtime, int recorded, int found)
{
  std::string msg = std::string("Inconsistent number of ") + runtime +
      " devices with the previous record! An earlier device manager recorded " + std::to_string(recorded) +
      " device(s) and this one found " + std::to_string(found) + ".";
  if (found == 0)
    msg += " A count of zero means this runtime found no usable device: check that the binary carries code for the"
           " architecture of the node it is running on.";
  return msg;
}

/** message for two device managers assigning a rank to different devices */
inline std::string deviceAssignmentMismatchMessage(const char* runtime,
                                                   int recorded,
                                                   int assigned,
                                                   int num_devices,
                                                   int local_rank,
                                                   int local_size)
{
  return std::string("Inconsistent assigned ") + runtime + " devices with the previous record! An earlier device" +
      " manager assigned device " + std::to_string(recorded) + " and this one assigned " + std::to_string(assigned) +
      " out of " + std::to_string(num_devices) + " visible, for local rank " + std::to_string(local_rank) + " of " +
      std::to_string(local_size) + ".";
}

/** distribute MPI ranks among devices
 *
 * the amount of MPI ranks for each device differs by 1 at maximum.
 * larger id has more MPI ranks.
 */
inline int determineDefaultDeviceNum(int num_devices, int rank_id, int num_ranks)
{
  if (num_ranks < num_devices)
    num_devices = num_ranks;
  // ranks are equally distributed among devices
  int min_ranks_per_device = num_ranks / num_devices;
  int residual             = num_ranks % num_devices;
  int assigned_device_id;
  if (rank_id < min_ranks_per_device * (num_devices - residual))
    assigned_device_id = rank_id / min_ranks_per_device;
  else
    assigned_device_id = (rank_id + num_devices - residual) / (min_ranks_per_device + 1);
  return assigned_device_id;
}
}

#endif
