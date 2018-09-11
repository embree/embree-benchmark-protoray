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

#include "core/intersector_stream.h"
#include "embree_intersector.h"

namespace prt {

template <int streamSize>
class EmbreeIntersectorStream : public IntersectorStream<streamSize>, EmbreeIntersector
{
public:
    EmbreeIntersectorStream(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(RayStream<streamSize>& rays, HitStream<streamSize>& hits, int count, RayStats& stats, RayHint hint)
    {
        stats.rayCount += count;

        ALIGNED_CACHE int geomIDs[streamSize];
        for (int i = 0; i < count; i += simdSize)
            store(&geomIDs[i], vint(RTC_INVALID_GEOMETRY_ID));

        RTCRayHitNp erays;

        erays.ray.org_x = rays.getOrgX();
        erays.ray.org_y = rays.getOrgY();
        erays.ray.org_z = rays.getOrgZ();

        erays.ray.dir_x = rays.getDirX();
        erays.ray.dir_y = rays.getDirY();
        erays.ray.dir_z = rays.getDirZ();

        erays.ray.tnear = 0;
        erays.ray.tfar = rays.getFar();

        erays.ray.time = 0;
        erays.ray.mask = 0;
        erays.ray.id = 0;
        erays.ray.flags = 0;

        erays.hit.Ng_x = 0;
        erays.hit.Ng_y = 0;
        erays.hit.Ng_z = 0;

        erays.hit.u = hits.getU();
        erays.hit.v = hits.getV();

        erays.hit.geomID = (unsigned int*)geomIDs;
        erays.hit.primID = (unsigned int*)hits.getPrimId();
        erays.hit.instID[0] = 0;

        RTCIntersectContext context;
        initIntersectContext(context, hint);
        rtcIntersectNp(scene, &context, &erays, count);
    }

    void occluded(RayStream<streamSize>& rays, int count, RayStats& stats, RayHint hint)
    {
        stats.rayCount += count;

        RTCRayNp erays;

        erays.org_x = rays.getOrgX();
        erays.org_y = rays.getOrgY();
        erays.org_z = rays.getOrgZ();

        erays.dir_x = rays.getDirX();
        erays.dir_y = rays.getDirY();
        erays.dir_z = rays.getDirZ();

        erays.tnear = 0;
        erays.tfar = rays.getFar();

        erays.time = 0;
        erays.mask = 0;
        erays.id = 0;
        erays.flags = 0;

        RTCIntersectContext context;
        initIntersectContext(context, hint);
        rtcOccludedNp(scene, &context, &erays, count);

        for (int i = 0; i < count; i += simdSize)
        {
            vfloat far = rays.far.getA(i);
            far = select(far < vfloat(zero), vfloat(zero), far);
            rays.far.setA(i, far);
        }
    }
};

template <int streamSize>
class EmbreeIntersectorSingleStream : public IntersectorStream<streamSize>, EmbreeIntersector
{
public:
    EmbreeIntersectorSingleStream(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(RayStream<streamSize>& rays, HitStream<streamSize>& hits, int count, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += count;

        for (int i = 0; i < count; ++i)
        {
            RTCRayHit eray;
            initRay(rays, i, eray.ray);
            initHit(eray.hit);
            rtcIntersect1(scene, &context, &eray);

            rays.far[i] = eray.ray.tfar;
            hits.primId[i] = eray.hit.primID;
            hits.u[i] = eray.hit.u;
            hits.v[i] = eray.hit.v;
        }
    }

    void occluded(RayStream<streamSize>& rays, int count, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += count;

        for (int i = 0; i < count; ++i)
        {
            RTCRay eray;
            initRay(rays, i, eray);
            rtcOccluded1(scene, &context, &eray);

            if (eray.tfar < 0.f)
                rays.far[i] = 0.f;
        }
    }
};

template <int streamSize>
class EmbreeIntersectorPacketStream : public IntersectorStream<streamSize>, EmbreeIntersector
{
public:
    EmbreeIntersectorPacketStream(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(RayStream<streamSize>& rays, HitStream<streamSize>& hits, int count, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += count;

        for (int i = 0; i < count; i += simdSize)
        {
            vbool mask = (vint(step) + i) < count;
            RaySimd ray;
            rays.getA(i, ray);

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

            rays.far.setA(i, load(eray.ray.tfar));
            hits.primId.setA(i, load((int*)eray.hit.primID));
            hits.u.setA(i, load(eray.hit.u));
            hits.v.setA(i, load(eray.hit.v));
        }
    }

    void occluded(RayStream<streamSize>& rays, int count, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += count;

        for (int i = 0; i < count; i += simdSize)
        {
            vbool mask = (vint(step) + i) < count;
            RaySimd ray;
            rays.getA(i, ray);

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

            ray.far = load(eray.tfar);
            ray.far = select(ray.far < vfloat(zero), vfloat(zero), ray.far);
            rays.far.setA(i, ray.far);
        }
    }
};

// SOA intersector only for testing
template <int streamSize>
class EmbreeIntersectorStreamSoa : public IntersectorStream<streamSize>, EmbreeIntersector
{
public:
    EmbreeIntersectorStreamSoa(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(RayStream<streamSize>& rays, HitStream<streamSize>& hits, int count, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        RTCRayHitNt<simdSize> erays[streamSize/simdSize];
        stats.rayCount += count;

        for (int i = 0; i < count; i += simdSize)
        {
            vbool mask = (vint(step) + i) < count;
            RaySimd ray;
            rays.getA(i, ray);
            ray.far = select(mask, ray.far, negInf);

            RTCRayHitNt<simdSize>& eray = erays[i/simdSize];
            initRay(ray, eray.ray);
            initHit(eray.hit);
        }

        rtcIntersectNM(scene, &context, (RTCRayHitN*)erays, simdSize, (count+simdSize-1)/simdSize, sizeof(RTCRayHitNt<simdSize>));

        for (int i = 0; i < count; i += simdSize)
        {
            const RTCRayHitNt<simdSize>& eray = erays[i/simdSize];

            rays.far.setA(i, load(eray.ray.tfar));
            hits.primId.setA(i, load((int*)eray.hit.primID));
            hits.u.setA(i, load(eray.hit.u));
            hits.v.setA(i, load(eray.hit.v));
        }
    }

    void occluded(RayStream<streamSize>& rays, int count, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        RTCRayNt<simdSize> erays[streamSize/simdSize];
        stats.rayCount += count;

        for (int i = 0; i < count; i += simdSize)
        {
            vbool mask = (vint(step) + i) < count;
            RaySimd ray;
            rays.getA(i, ray);
            ray.far = select(mask, ray.far, negInf);

            RTCRayNt<simdSize>& eray = erays[i/simdSize];
            initRay(ray, eray);
        }

        rtcOccludedNM(scene, &context, (RTCRayN*)erays, simdSize, (count+simdSize-1)/simdSize, sizeof(RTCRayNt<simdSize>));

        for (int i = 0; i < count; i += simdSize)
        {
            const RTCRayNt<simdSize>& eray = erays[i/simdSize];

            vfloat far = load(eray.tfar);
            far = select(far < vfloat(zero), vfloat(zero), far);
            rays.far.setA(i, far);
        }
    }
};

} // namespace prt
