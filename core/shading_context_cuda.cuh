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

#include "math/math.cuh"

namespace prt {

struct ShadingContextCuda
{
    float3 p;        // position
    Basis3f f;       // shading frame
    float3 Ng;       // geometric normal
    float2 uv;       // texture coords
    bool backfacing; // is the geometry backfacing?
    float eps;       // intersection epsilon

    CUDA_DEV_FORCEINLINE const Basis3f& getFrame() const
    {
        return f;
    }

    CUDA_DEV_FORCEINLINE float3 getN() const
    {
        return f.N;
    }
};

struct SimpleShadingContextCuda
{
    float3 p;        // position
    float3 Ng;       // geometric normal
    float eps;       // intersection epsilon

    CUDA_DEV_FORCEINLINE Basis3f getFrame() const
    {
        return makeFrame(Ng);
    }

    CUDA_DEV_FORCEINLINE float3 getN() const
    {
        return Ng;
    }
};

} // namespace prt
