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

#include "core/intersector_single.h"
#include "core/intersector_packet.h"
#include "core/intersector_stream.h"
#include "embree_intersector.h"

namespace prt {

class EmbreeIntersectorSingle : public IntersectorSingle, EmbreeIntersector
{
public:
    EmbreeIntersectorSingle(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(Ray& ray, Hit& hit, RayStats& stats)
    {

        RTCRay eray;
        stats.rayCount++;
        initRay(ray, eray);
        rtcIntersect(scene, eray);

        ray.far = eray.tfar;
        hit.primId = eray.primID;
        hit.u = eray.u;
        hit.v = eray.v;
    }

    void occluded(Ray& ray, RayStats& stats)
    {

        stats.rayCount++;

        RTCRay eray;
        initRay(ray, eray);
        rtcOccluded(scene, eray);

        if (eray.geomID == 0)
            ray.far = 0.0f;
    }

private:
    FORCEINLINE void initRay(const Ray& ray, RTCRay& eray)
    {
        eray.org[0] = ray.org[0];
        eray.org[1] = ray.org[1];
        eray.org[2] = ray.org[2];

        eray.dir[0] = ray.dir[0];
        eray.dir[1] = ray.dir[1];
        eray.dir[2] = ray.dir[2];

        eray.tnear = 0.0f;
        eray.tfar = ray.far;

        eray.geomID = RTC_INVALID_GEOMETRY_ID;
        eray.primID = RTC_INVALID_GEOMETRY_ID;
        //eray.instID = RTC_INVALID_GEOMETRY_ID;

        eray.mask = -1;
        eray.time = 0.0f;
    }
};

class EmbreeIntersectorSinglePacket : public IntersectorPacket, EmbreeIntersector
{
public:
    EmbreeIntersectorSinglePacket(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(vbool mask, RaySimd& ray, HitSimd& hit, RayStats& stats)
    {
        int intMask = toIntMask(mask);
        stats.rayCount += bitCount(intMask);

        int i = -1;
        while ((i = bitScan(intMask, i)) < simdSize)
        {
            RTCRay eray;
            initRay(ray, i, eray);
            rtcIntersect(scene, eray);

            ray.far[i] = eray.tfar;
            hit.primId[i] = eray.primID;
            hit.u[i] = eray.u;
            hit.v[i] = eray.v;
        }
    }

    void occluded(vbool mask, RaySimd& ray, RayStats& stats)
    {
        int intMask = toIntMask(mask);
        stats.rayCount += bitCount(intMask);

        int i = -1;
        while ((i = bitScan(intMask, i)) < simdSize)
        {
            RTCRay eray;
            initRay(ray, i, eray);
            rtcOccluded(scene, eray);

            if (eray.geomID == 0)
                ray.far[i] = 0.0f;
        }
    }

private:
    FORCEINLINE void initRay(const RaySimd& ray, int i, RTCRay& eray)
    {
        eray.org[0] = ray.org.x[i];
        eray.org[1] = ray.org.y[i];
        eray.org[2] = ray.org.z[i];

        eray.dir[0] = ray.dir.x[i];
        eray.dir[1] = ray.dir.y[i];
        eray.dir[2] = ray.dir.z[i];

        eray.tnear = 0.0f;
        eray.tfar = ray.far[i];

        eray.geomID = RTC_INVALID_GEOMETRY_ID;
        eray.primID = RTC_INVALID_GEOMETRY_ID;
        //eray.instID = RTC_INVALID_GEOMETRY_ID;

        eray.mask = -1;
        eray.time = 0.0f;
    }
};

template <int streamSize>
class EmbreeIntersectorSingleStream : public IntersectorStream<streamSize>, EmbreeIntersector
{
public:
    EmbreeIntersectorSingleStream(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(RayStream<streamSize>& rays, HitStream<streamSize>& hits, int count, RayStats& stats, RayHint hint)
    {
        stats.rayCount += count;

        for (int i = 0; i < count; ++i)
        {
            RTCRay eray;
            initRay(rays, i, eray);
            rtcIntersect(scene, eray);

            rays.far[i] = eray.tfar;
            hits.primId[i] = eray.primID;
            hits.u[i] = eray.u;
            hits.v[i] = eray.v;
        }
    }

    void occluded(RayStream<streamSize>& rays, int count, RayStats& stats, RayHint hint)
    {
        stats.rayCount += count;

        for (int i = 0; i < count; ++i)
        {
            RTCRay eray;
            initRay(rays, i, eray);
            rtcOccluded(scene, eray);

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

} // namespace prt
