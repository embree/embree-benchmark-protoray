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

struct HitCuda
{
    float t;
    int primId;
    float u;
    float v;

    CUDA_DEV_FORCEINLINE bool isHit() const
    {
        return t >= 0.0f;
    }
};

struct AnyHitCuda
{
    float t;

    CUDA_DEV_FORCEINLINE bool isHit() const
    {
        return t >= 0.0f;
    }
};

struct RayCuda
{
    float3 org;
    float near;
    float3 dir;
    float far;

    CUDA_DEV_FORCEINLINE void init(const float3& org, const float3& dir)
    {
        this->org = org;
        this->near = 0.0f;
        this->dir = dir;
        this->far = FLT_MAX;
    }

    CUDA_DEV_FORCEINLINE void init(const float3& org, const float3& dir, float near)
    {
        this->org = org;
        this->near = near;
        this->dir = dir;
        this->far = FLT_MAX;
    }

    CUDA_DEV_FORCEINLINE float3 getHitPoint(const HitCuda& hit) const
    {
        return org + hit.t * dir;
    }

    CUDA_DEV_FORCEINLINE float3 getHitPoint(const HitCuda& hit, float& eps) const
    {
        float3 p = getHitPoint(hit);
        eps = max(hit.t, reduceMax(abs(p))) * 0x1.fp-18;
        return p;
    }
};

} // namespace prt
