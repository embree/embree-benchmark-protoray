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

#include "sys/memory.h"
#include "ray.h"
#include "ray_embree.h"

namespace prt {

template <class T, int size>
class RayStreamChannelAos
{
private:
    ALIGNED_CACHE T v[size];

public:
    FORCEINLINE const T& operator [](size_t i) const
    {
        return v[i];
    }

    FORCEINLINE T& operator [](size_t i)
    {
        return v[i];
    }

    FORCEINLINE const T* get() const { return v; }
    FORCEINLINE T* get() { return v; }
};

template <int size>
class RayStreamAos
{
private:
    ALIGNED_CACHE RTCRay v[size];

public:
    FORCEINLINE void set(int i, const Ray& ray)
    {
        v[i].org_x = ray.org.x;
        v[i].org_y = ray.org.y;
        v[i].org_z = ray.org.z;

        v[i].dir_x = ray.dir.x;
        v[i].dir_y = ray.dir.y;
        v[i].dir_z = ray.dir.z;

        v[i].tnear = 0.0f;
        v[i].tfar = ray.far;

        v[i].mask = -1;
        v[i].time = 0.0f;
    }

    FORCEINLINE void getRay(int i, Ray& ray)
    {
        ray.org.x = v[i].org_x;
        ray.org.y = v[i].org_y;
        ray.org.z = v[i].org_z;

        ray.dir.x = v[i].dir_x;
        ray.dir.y = v[i].dir_y;
        ray.dir.z = v[i].dir_z;

        ray.far = v[i].tfar;
    }

    FORCEINLINE bool isHit(int i) const
    {
        return v[i].tfar < float(posMax);
    }

    FORCEINLINE bool isOccluded(int i) const
    {
        return v[i].tfar < 0.f;
    }

    FORCEINLINE bool isNotOccluded(int i) const
    {
        return v[i].tfar >= 0.f;
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

template <int size>
class RayHitStreamAos
{
private:
    ALIGNED_CACHE RTCRayHit v[size];

public:
    FORCEINLINE void set(int i, const Ray& ray)
    {
        v[i].ray.org_x = ray.org.x;
        v[i].ray.org_y = ray.org.y;
        v[i].ray.org_z = ray.org.z;

        v[i].ray.dir_x = ray.dir.x;
        v[i].ray.dir_y = ray.dir.y;
        v[i].ray.dir_z = ray.dir.z;

        v[i].ray.tnear = 0.0f;
        v[i].ray.tfar = ray.far;

        v[i].ray.mask = -1;
        v[i].ray.time = 0.0f;

        v[i].hit.geomID = RTC_INVALID_GEOMETRY_ID;
    }

    FORCEINLINE void getRay(int i, Ray& ray)
    {
        ray.org.x = v[i].ray.org_x;
        ray.org.y = v[i].ray.org_y;
        ray.org.z = v[i].ray.org_z;

        ray.dir.x = v[i].ray.dir_x;
        ray.dir.y = v[i].ray.dir_y;
        ray.dir.z = v[i].ray.dir_z;

        ray.far = v[i].ray.tfar;
    }

    FORCEINLINE void getHit(int i, Hit& hit)
    {
        hit.primId = v[i].hit.primID;
        hit.u = v[i].hit.u;
        hit.v = v[i].hit.v;
    }

    FORCEINLINE bool isHit(int i) const
    {
        return v[i].ray.tfar < float(posMax);
    }

    FORCEINLINE bool isOccluded(int i) const
    {
        return v[i].ray.tfar < 0.f;
    }

    FORCEINLINE bool isNotOccluded(int i) const
    {
        return v[i].ray.tfar >= 0.f;
    }

    FORCEINLINE const RTCRayHit& operator [](size_t i) const
    {
        return v[i];
    }

    FORCEINLINE RTCRayHit& operator [](size_t i)
    {
        return v[i];
    }

    FORCEINLINE RTCRayHit* get() { return v; }
    FORCEINLINE const RTCRayHit* get() const { return v; }
};

} // namespace prt
