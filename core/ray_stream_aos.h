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

#include "sys/memory.h"
#include "ray_embree.h"

namespace prt {

template <int size>
class RayHitStreamAos
{
private:
    ALIGNED_CACHE RTCRay v[size];

public:
    FORCEINLINE void set(int i, const Ray& ray)
    {
        v[i].org[0] = ray.org.x;
        v[i].org[1] = ray.org.y;
        v[i].org[2] = ray.org.z;

        v[i].dir[0] = ray.dir.x;
        v[i].dir[1] = ray.dir.y;
        v[i].dir[2] = ray.dir.z;

        v[i].tnear = 0.0f;
        v[i].tfar = ray.far;

        v[i].geomID = RTC_INVALID_GEOMETRY_ID;
        v[i].primID = RTC_INVALID_GEOMETRY_ID;
        //v[i].instID = RTC_INVALID_GEOMETRY_ID;

        v[i].mask = -1;
        v[i].time = 0.0f;
    }

    FORCEINLINE void getRay(int i, Ray& ray)
    {
        ray.org.x = v[i].org[0];
        ray.org.y = v[i].org[1];
        ray.org.z = v[i].org[2];

        ray.dir.x = v[i].dir[0];
        ray.dir.y = v[i].dir[1];
        ray.dir.z = v[i].dir[2];

        ray.far = v[i].tfar;
    }

    FORCEINLINE void getHit(int i, Hit& hit)
    {
        hit.primId = v[i].primID;
        hit.u = v[i].u;
        hit.v = v[i].v;
    }

    FORCEINLINE bool isHit(int i) const
    {
        return v[i].tfar < float(posMax);
    }

    FORCEINLINE bool isOccluded(int i) const
    {
        return v[i].geomID == 0;
    }

    FORCEINLINE bool isNotOccluded(int i) const
    {
        return v[i].geomID != 0;
    }

    FORCEINLINE const RTCRay& operator [](size_t i) const
    {
        return v[i];
    }

    FORCEINLINE RTCRay& operator [](size_t i)
    {
        return v[i];
    }

    FORCEINLINE RTCRay* get() { return v; }
    FORCEINLINE const RTCRay* get() const { return v; }
};

} // namespace prt
