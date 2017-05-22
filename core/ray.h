// ======================================================================== //
// Copyright 2015-2017 Intel Corporation                                    //
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

#include "sys/common.h"
#include "math/vec2.h"
#include "math/vec3.h"
#include "math/basis3.h"

namespace prt {

struct Ray
{
    Vec3f org; // origin
    Vec3f dir; // direction
    float far; // hit distance

    FORCEINLINE float* getOrgX() { return &org.x; }
    FORCEINLINE float* getOrgY() { return &org.y; }
    FORCEINLINE float* getOrgZ() { return &org.z; }
    FORCEINLINE float* getDirX() { return &dir.x; }
    FORCEINLINE float* getDirY() { return &dir.y; }
    FORCEINLINE float* getDirZ() { return &dir.z; }
    FORCEINLINE float* getFar()  { return &far; }

    FORCEINLINE void init(const Vec3f& org, const Vec3f& dir)
    {
        this->org = org;
        this->dir = dir;
        this->far = posMax;
    }

    FORCEINLINE void init(const Vec3f& org, const Vec3f& dir, float near)
    {
        this->org = org + near * dir;
        this->dir = dir;
        this->far = posMax;
    }

    FORCEINLINE void init(const Vec3f& org, const Vec3f& dir, float near, float far)
    {
        this->org = org + near * dir;
        this->dir = dir;
        this->far = far - near;
    }

    FORCEINLINE bool isHit() const
    {
        return far < float(posMax);
    }

    FORCEINLINE Vec3f getHitPoint() const
    {
        return org + far * dir;
    }

    FORCEINLINE Vec3f getHitPoint(float& eps) const
    {
        Vec3f p = getHitPoint();
        eps = max(far, reduceMax(abs(p))) * 0x1.fp-18;
        return p;
    }

    FORCEINLINE bool isOccluded() const
    {
        return far == 0.0f;
    }

    FORCEINLINE bool isNotOccluded() const
    {
        return far > 0.0f;
    }
};

struct Hit
{
    int primId; // primitive ID
    float u;    // u coord
    float v;    // v coord

    FORCEINLINE int*   getPrimId() { return &primId; }
    FORCEINLINE float* getU()      { return &u; }
    FORCEINLINE float* getV()      { return &v; }
};

} // namespace prt
