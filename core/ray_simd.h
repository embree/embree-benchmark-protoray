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
#include "math/simd.h"
#include "math/vec3.h"
#include "math/basis3.h"

namespace prt {

struct RaySimd
{
    Vec3vf org; // origin
    Vec3vf dir; // direction
    vfloat far; // hit distance

    FORCEINLINE float* getOrgX() { return &org.x[0]; }
    FORCEINLINE float* getOrgY() { return &org.y[0]; }
    FORCEINLINE float* getOrgZ() { return &org.z[0]; }
    FORCEINLINE float* getDirX() { return &dir.x[0]; }
    FORCEINLINE float* getDirY() { return &dir.y[0]; }
    FORCEINLINE float* getDirZ() { return &dir.z[0]; }
    FORCEINLINE float* getFar()  { return &far[0]; }

    FORCEINLINE void init(const Vec3vf& org, const Vec3vf& dir)
    {
        this->org = org;
        this->dir = dir;
        this->far = posMax;
    }

    FORCEINLINE void init(const Vec3vf& org, const Vec3vf& dir, vfloat near)
    {
        this->org = org + near * dir;
        this->dir = dir;
        this->far = posMax;
    }

    FORCEINLINE void init(const Vec3vf& org, const Vec3vf& dir, vfloat near, vfloat far)
    {
        this->org = org + near * dir;
        this->dir = dir;
        this->far = far - near;
    }

    FORCEINLINE vbool isHit() const
    {
        return far < float(posMax);
    }

    FORCEINLINE Vec3vf getHitPoint() const
    {
        return org + far * dir;
    }

    FORCEINLINE Vec3vf getHitPoint(vfloat& eps) const
    {
        Vec3vf p = getHitPoint();
        eps = max(far, reduceMax(abs(p))) * 0x1.fp-18;
        return p;
    }

    FORCEINLINE vbool isOccluded() const
    {
        return far == zero;
    }

    FORCEINLINE vbool isNotOccluded() const
    {
        return far > zero;
    }
};

struct HitSimd
{
    vint primId; // primitive ID
    vfloat u;    // u coord
    vfloat v;    // v coord

    FORCEINLINE int*   getPrimId() { return &primId[0]; }
    FORCEINLINE float* getU()      { return &u[0]; }
    FORCEINLINE float* getV()      { return &v[0]; }
};

} // namespace prt
