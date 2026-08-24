//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: Rohit Goswami, rgoswami@ieee.org, University of Iceland
//
// File created by: Rohit Goswami, rgoswami@ieee.org, University of Iceland
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_CUDA_GRAPH_H
#define QMCPLUSPLUS_CUDA_GRAPH_H

#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include "CUDAruntime.hpp"

namespace qmcplusplus
{
namespace compute
{

/** a recorded sequence of stream operations, replayed as one launch
 *
 *  A particle-by-particle step issues the same short sequence of copies and kernels for
 *  every electron, and the driver charges for each of them separately. A graph is that
 *  sequence recorded once and submitted as a unit.
 *
 *  What a graph bakes in is the kernel arguments and the grid shapes. What it does not
 *  bake in is the contents of the memory its nodes read: a recorded host to device copy
 *  reads its source when the graph is launched, so one recording serves every electron
 *  whose arguments and shapes match. The caller says which those are by handing over a
 *  key; anything that appears as a kernel argument or a grid bound has to be in it,
 *  including the device addresses of the buffers, so that a reallocation produces a new
 *  key rather than a graph pointing at freed memory.
 *
 *  A sequence is run normally the first time a key is seen and recorded the second time.
 *  Any one-off setup inside it therefore happens outside the recording, which is what
 *  keeps a lazily allocating callee from being captured mid-allocation: capture rejects
 *  allocation outright.
 *
 *  One cache belongs to one stream, which in practice means one crowd, so nothing here is
 *  shared between threads. Capture is still asked for in thread-local mode, because
 *  another crowd may be recording on its own stream at the same time.
 */
class CUDAGraphCache
{
public:
  CUDAGraphCache() = default;
  CUDAGraphCache(const CUDAGraphCache&) = delete;
  CUDAGraphCache& operator=(const CUDAGraphCache&) = delete;

  ~CUDAGraphCache()
  {
    for (auto& [key, exec] : execs_)
      cudaGraphExecDestroy(exec);
  }

  /** run fn, as a recording of it where one is to hand
   *
   *  @param stream the stream fn issues its work on
   *  @param key    everything fn's arguments and grid shapes depend on
   */
  template<class F>
  void launch(cudaStream_t stream, uint64_t key, F&& fn)
  {
    if (!enabled())
    {
      fn();
      return;
    }

    if (const auto it = execs_.find(key); it != execs_.end())
    {
      cudaErrorCheck(cudaGraphLaunch(it->second, stream), "cudaGraphLaunch failed!");
      return;
    }

    if (seen_.insert(key).second)
    {
      // first sighting: run it, so that whatever it does once is not inside the recording
      fn();
      return;
    }

    cudaGraph_t graph;
    cudaErrorCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                   "cudaStreamBeginCapture failed!");
    fn();
    cudaErrorCheck(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture failed!");

    cudaGraphExec_t exec;
    cudaErrorCheck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "cudaGraphInstantiate failed!");
    cudaErrorCheck(cudaGraphDestroy(graph), "cudaGraphDestroy failed!");
    execs_.emplace(key, exec);
    cudaErrorCheck(cudaGraphLaunch(exec, stream), "cudaGraphLaunch failed!");
  }

  /// fold one value into a key
  static uint64_t mix(uint64_t key, uint64_t value)
  {
    // splitmix64's finalizer, which is cheap and spreads a pointer's low bits
    key ^= value + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);
    key ^= key >> 30;
    key *= 0xbf58476d1ce4e5b9ULL;
    key ^= key >> 27;
    return key;
  }

  static uint64_t mix(uint64_t key, const void* pointer) { return mix(key, reinterpret_cast<uint64_t>(pointer)); }

  /// how many distinct sequences are recorded, for a caller that wants to bound it
  size_t size() const { return execs_.size(); }

  static bool enabled()
  {
    static const bool on = [] {
      const char* off = std::getenv("QMCPACK_CUDA_GRAPHS");
      return !(off && *off == '0');
    }();
    return on;
  }

private:
  std::unordered_map<uint64_t, cudaGraphExec_t> execs_;
  /// keys met once; a sequence is recorded on its second sighting, not its first
  std::unordered_set<uint64_t> seen_;
};

} // namespace compute
} // namespace qmcplusplus

#endif
