// ======================================================================== //
// Copyright 2015-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define CUDA_DEV __device__
#define CUDA_DEV_FORCEINLINE __device__ __inline__
#define CUDA_HOST_DEV __device__ __host__
#define CUDA_HOST_DEV_FORCEINLINE __device__ __host__ __inline__
#define CUDA_DEV_KERNEL __global__

namespace prt {

typedef unsigned int uint;

template <typename T>
CUDA_HOST_DEV_FORCEINLINE void swap(T& a, T& b)
{
    T t = a;
    a = b;
    b = t;
}

template <typename T>
CUDA_DEV_FORCEINLINE T loadConst(const T* ptr)
{
#if __CUDA_ARCH__ < 350
    return *ptr;
#else
    return __ldg(ptr);
#endif
}

CUDA_DEV_FORCEINLINE int laneId()
{
    return threadIdx.x % 32;
}

// Warp-aggregated atomic increment
CUDA_DEV_FORCEINLINE int atomicIncAgg(int* ctr)
{
    auto g = cooperative_groups::coalesced_threads();
    int warp_res;
    if (g.thread_rank() == 0)
        warp_res = atomicAdd(ctr, g.size());
    return g.shfl(warp_res, 0) + g.thread_rank();
}

} // namespace prt
