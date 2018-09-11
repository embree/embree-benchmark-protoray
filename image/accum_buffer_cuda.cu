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

#include "math/math.cuh"
#include "accum_buffer_cuda.h"

namespace prt {

CUDA_DEV_KERNEL void accumBufferUpdateKernel(AccumBufferCuda accumBuffer, int* colorBuffer)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float4 accum = accumBuffer.data[x + y * accumBuffer.size.x];

    float w = rcp(accum.w);
    const float gamma = 1.0f/2.2f;

    int r = clamp(int(pow(accum.x * w, gamma) * 255.0f), 0, 255);
    int g = clamp(int(pow(accum.y * w, gamma) * 255.0f), 0, 255);
    int b = clamp(int(pow(accum.z * w, gamma) * 255.0f), 0, 255);

    colorBuffer[x + y * accumBuffer.size.x] = b | (g << 8) | (r << 16);
}

void AccumBufferCuda::update(void* colorBuffer)
{
    dim3 blockSize(32, 4);
    dim3 gridSize(size.x / blockSize.x, size.y / blockSize.y);
    accumBufferUpdateKernel<<<gridSize, blockSize>>>(*this, (int*)colorBuffer);
}

} // namespace prt
