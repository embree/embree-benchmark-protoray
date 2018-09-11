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

#include "core/intersector_packet.h"
#include "embree_intersector.h"

namespace prt {

class EmbreeIntersectorPacket : public IntersectorPacket, EmbreeIntersector
{
public:
    EmbreeIntersectorPacket(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(vbool mask, RaySimd& ray, HitSimd& hit, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += bitCount(toIntMask(mask));

#if SIMD_SIZE == 16
        vint16 emask = select(mask, vint16(-1), zero);
        RTCRayHit16 eray;
        initRay(ray, eray.ray);
        initHit(eray.hit);
        rtcIntersect16((const int*)&emask, scene, &context, &eray);
#else
        RTCRayHit8 eray;
        initRay(ray, eray.ray);
        initHit(eray.hit);
        rtcIntersect8((const int*)&mask, scene, &context, &eray);
#endif

        ray.far = load(eray.ray.tfar);
        hit.primId = load((int*)eray.hit.primID);
        hit.u = load(eray.hit.u);
        hit.v = load(eray.hit.v);
    }

    void occluded(vbool mask, RaySimd& ray, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += bitCount(toIntMask(mask));

#if SIMD_SIZE == 16
        vint16 emask = select(mask, vint16(-1), zero);
        RTCRay16 eray;
        initRay(ray, eray);
        rtcOccluded16((const int*)&emask, scene, &context, &eray);
#else
        RTCRay8 eray;
        initRay(ray, eray);
        rtcOccluded8((const int*)&mask, scene, &context, &eray);
#endif

        vbool hitMask = load(eray.tfar) < vfloat(zero);
        set(hitMask, ray.far, vfloat(zero));
    }
};

#if SIMD_SIZE == 16
class EmbreeIntersectorPacket8 : public IntersectorPacket, EmbreeIntersector
{
public:
    EmbreeIntersectorPacket8(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(vbool mask, RaySimd& ray, HitSimd& hit, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += bitCount(toIntMask(mask));

        vint emask = select(mask, vint(-1), zero);
        for (size_t i = 0; i < simdSize; i += 8)
        {
            RTCRayHit8 eray;
            initRay(ray, i, eray.ray);
            initHit(eray.hit);
            rtcIntersect8((const int*)&emask[i], scene, &context, &eray);

            store(&ray.far[i],    load<8>(eray.ray.tfar));
            store(&hit.primId[i], load<8>((int*)eray.hit.primID));
            store(&hit.u[i],      load<8>(eray.hit.u));
            store(&hit.v[i],      load<8>(eray.hit.v));
        }
    }

    void occluded(vbool mask, RaySimd& ray, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += bitCount(toIntMask(mask));

        vint emask = select(mask, vint(-1), zero);
        for (size_t i = 0; i < simdSize; i += 8)
        {
            RTCRay8 eray;
            initRay(ray, i, eray);
            rtcOccluded8((const int*)&emask[i], scene, &context, &eray);

            vbool8 hitMask = load<8>(eray.tfar) < vfloat8(zero);
            store(hitMask, &ray.far[i], vfloat8(zero));
        }
    }
};
#elif !defined(__MIC__)
typedef EmbreeIntersectorPacket EmbreeIntersectorPacket8;
#endif

class EmbreeIntersectorSinglePacket : public IntersectorPacket, EmbreeIntersector
{
public:
    EmbreeIntersectorSinglePacket(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(vbool mask, RaySimd& ray, HitSimd& hit, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        int intMask = toIntMask(mask);
        stats.rayCount += bitCount(intMask);

        int i = -1;
        while ((i = bitScan(intMask, i)) < simdSize)
        {
            RTCRayHit eray;
            initRay(ray, i, eray.ray);
            initHit(eray.hit);
            rtcIntersect1(scene, &context, &eray);

            ray.far[i] = eray.ray.tfar;
            hit.primId[i] = eray.hit.primID;
            hit.u[i] = eray.hit.u;
            hit.v[i] = eray.hit.v;
        }
    }

    void occluded(vbool mask, RaySimd& ray, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        int intMask = toIntMask(mask);
        stats.rayCount += bitCount(intMask);

        int i = -1;
        while ((i = bitScan(intMask, i)) < simdSize)
        {
            RTCRay eray;
            initRay(ray, i, eray);
            rtcOccluded1(scene, &context, &eray);

            if (eray.tfar < 0.f)
                ray.far[i] = 0.f;
        }
    }
};

} // namespace prt
