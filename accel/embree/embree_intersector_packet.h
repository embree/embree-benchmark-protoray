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

#include "core/intersector_packet.h"
#include "embree_intersector.h"

namespace prt {

class EmbreeIntersectorPacket : public IntersectorPacket, EmbreeIntersector
{
public:
    EmbreeIntersectorPacket(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(vbool mask, RaySimd& ray, HitSimd& hit, RayStats& stats)
    {
#if SIMD_SIZE == 16
        vint16 emask = select(mask, vint16(-1), zero);
        RTCRay16 eray;
        stats.rayCount += bitCount(toIntMask(mask));
        initRay(ray, eray);
        rtcIntersect16(&emask, scene, eray);
#else
        RTCRay8 eray;
        initRay(ray, eray);
        rtcIntersect8(&mask, scene, eray);
#endif

        ray.far = load(eray.tfar);
        hit.primId = load((int*)eray.primID);
        hit.u = load(eray.u);
        hit.v = load(eray.v);
    }

    void occluded(vbool mask, RaySimd& ray, RayStats& stats)
    {
        stats.rayCount += bitCount(toIntMask(mask));

#if SIMD_SIZE == 16
        vint16 emask = select(mask, vint16(-1), zero);
        RTCRay16 eray;
        initRay(ray, eray);
        rtcOccluded16(&emask, scene, eray);
#else
        RTCRay8 eray;
        initRay(ray, eray);
        rtcOccluded8(&mask, scene, eray);
#endif

        vbool hitMask = load((int*)eray.geomID) == vint(zero);
        set(hitMask, ray.far, vfloat(zero));
    }

private:
    template <class RTCRayT>
    FORCEINLINE void initRay(const RaySimd& ray, RTCRayT& eray)
    {
        store(eray.orgx, ray.org.x);
        store(eray.orgy, ray.org.y);
        store(eray.orgz, ray.org.z);

        store(eray.dirx, ray.dir.x);
        store(eray.diry, ray.dir.y);
        store(eray.dirz, ray.dir.z);

        store(eray.tnear, vfloat(zero));
        store(eray.tfar, ray.far);

        store((int*)eray.geomID, vint(RTC_INVALID_GEOMETRY_ID));
        store((int*)eray.primID, vint(RTC_INVALID_GEOMETRY_ID));
        //store((int*)eray.instID, vint(RTC_INVALID_GEOMETRY_ID));

        store((int*)eray.mask, vint(-1));
        store(eray.time, vfloat(zero));
    }
};

#if SIMD_SIZE == 16
class EmbreeIntersectorPacket8 : public IntersectorPacket, EmbreeIntersector
{
public:
    EmbreeIntersectorPacket8(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(vbool mask, RaySimd& ray, HitSimd& hit, RayStats& stats)
    {
        stats.rayCount += bitCount(toIntMask(mask));

        vint emask = select(mask, vint(-1), zero);
        for (size_t i = 0; i < simdSize; i += 8)
        {
            RTCRay8 eray;
            initRay(ray, i, eray);
            rtcIntersect8(&emask[i], scene, eray);

            store(&ray.far[i],    load<8>(eray.tfar));
            store(&hit.primId[i], load<8>((int*)eray.primID));
            store(&hit.u[i],      load<8>(eray.u));
            store(&hit.v[i],      load<8>(eray.v));
        }
    }

    void occluded(vbool mask, RaySimd& ray, RayStats& stats)
    {
        stats.rayCount += bitCount(toIntMask(mask));

        vint emask = select(mask, vint(-1), zero);
        for (size_t i = 0; i < simdSize; i += 8)
        {
            RTCRay8 eray;
            initRay(ray, i, eray);
            rtcOccluded8(&emask[i], scene, eray);

            vbool8 hitMask = load<8>((int*)eray.geomID) == vint8(zero);
            store(hitMask, &ray.far[i], vfloat8(zero));
        }
    }

private:
    FORCEINLINE void initRay(const RaySimd& ray, size_t i, RTCRay8& eray)
    {
        store(eray.orgx, load<8>(&ray.org.x[i]));
        store(eray.orgy, load<8>(&ray.org.y[i]));
        store(eray.orgz, load<8>(&ray.org.z[i]));

        store(eray.dirx, load<8>(&ray.dir.x[i]));
        store(eray.diry, load<8>(&ray.dir.y[i]));
        store(eray.dirz, load<8>(&ray.dir.z[i]));

        store(eray.tnear, vfloat8(zero));
        store(eray.tfar, load<8>(&ray.far[i]));

        store((int*)eray.geomID, vint8(RTC_INVALID_GEOMETRY_ID));
        store((int*)eray.primID, vint8(RTC_INVALID_GEOMETRY_ID));
        //store((int*)eray.instID, vint8(RTC_INVALID_GEOMETRY_ID));

        store((int*)eray.mask, vint8(-1));
        store(eray.time, vfloat8(zero));
    }
};
#elif !defined(__MIC__)
typedef EmbreeIntersectorPacket EmbreeIntersectorPacket8;
#endif

} // namespace prt
