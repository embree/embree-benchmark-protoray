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

        RTCRayNp erays;

        erays.orgx = rays.getOrgX();
        erays.orgy = rays.getOrgY();
        erays.orgz = rays.getOrgZ();

        erays.dirx = rays.getDirX();
        erays.diry = rays.getDirY();
        erays.dirz = rays.getDirZ();

        erays.tnear = 0;
        erays.tfar = rays.getFar();

        erays.time = 0;
        erays.mask = 0;

        erays.Ngx = 0;
        erays.Ngy = 0;
        erays.Ngz = 0;

        erays.u = hits.getU();
        erays.v = hits.getV();

        erays.geomID = (unsigned int*)geomIDs;
        erays.primID = (unsigned int*)hits.getPrimId();
        erays.instID = 0;

        RTCIntersectContext context;
        initIntersectContext(context, hint);
        rtcIntersectNp(scene, &context, erays, count);
    }

    void occluded(RayStream<streamSize>& rays, int count, RayStats& stats, RayHint hint)
    {
        stats.rayCount += count;

        ALIGNED_CACHE int geomIDs[streamSize];
        for (int i = 0; i < count; i += simdSize)
            store(&geomIDs[i], vint(RTC_INVALID_GEOMETRY_ID));

        RTCRayNp erays;

        erays.orgx = rays.getOrgX();
        erays.orgy = rays.getOrgY();
        erays.orgz = rays.getOrgZ();

        erays.dirx = rays.getDirX();
        erays.diry = rays.getDirY();
        erays.dirz = rays.getDirZ();

        erays.tnear = 0;
        erays.tfar = rays.getFar();

        erays.time = 0;
        erays.mask = 0;

        erays.Ngx = 0;
        erays.Ngy = 0;
        erays.Ngz = 0;

        erays.u = 0;
        erays.v = 0;

        erays.geomID = (unsigned int*)geomIDs;
        erays.primID = 0;
        erays.instID = 0;

        RTCIntersectContext context;
        initIntersectContext(context, hint);
        rtcOccludedNp(scene, &context, erays, count);

        for (int i = 0; i < count; i += simdSize)
            store(load(&geomIDs[i]) == vint(zero), &rays.far[i], vfloat(zero));
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
            RTCRay eray;
            initRay(rays, i, eray);
            rtcIntersect1Ex(scene, &context, eray);

            rays.far[i] = eray.tfar;
            hits.primId[i] = eray.primID;
            hits.u[i] = eray.u;
            hits.v[i] = eray.v;
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
            rtcOccluded1Ex(scene, &context, eray);

            if (eray.geomID == 0)
                rays.far[i] = 0.0f;
        }
    }

private:
    FORCEINLINE void initRay(const RayStream<streamSize>& rays, int i, RTCRay& eray)
    {
        eray.org[0] = rays.org.x[i];
        eray.org[1] = rays.org.y[i];
        eray.org[2] = rays.org.z[i];

        eray.dir[0] = rays.dir.x[i];
        eray.dir[1] = rays.dir.y[i];
        eray.dir[2] = rays.dir.z[i];

        eray.tnear = 0.0f;
        eray.tfar = rays.far[i];

        eray.geomID = RTC_INVALID_GEOMETRY_ID;
        eray.primID = RTC_INVALID_GEOMETRY_ID;
        //eray.instID = RTC_INVALID_GEOMETRY_ID;

        eray.mask = -1;
        eray.time = 0.0f;
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
            RTCRay16 eray;
            initRay(ray, eray);
            rtcIntersect16Ex(&emask, scene, &context, eray);
#else
            RTCRay8 eray;
            initRay(ray, eray);
            rtcIntersect8Ex(&mask, scene, &context, eray);
#endif

            rays.far.setA(i, load(eray.tfar));
            hits.primId.setA(i, load((int*)eray.primID));
            hits.u.setA(i, load(eray.u));
            hits.v.setA(i, load(eray.v));
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
            rtcOccluded16Ex(&emask, scene, &context, eray);
#else
            RTCRay8 eray;
            initRay(ray, eray);
            rtcOccluded8Ex(&mask, scene, &context, eray);
#endif

            vbool hitMask = load((int*)eray.geomID) == vint(zero);
            ray.far = select(hitMask, vfloat(zero), ray.far);
            rays.far.setA(i, ray.far);
        }
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

} // namespace prt
