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
        if (hint == rayHintCoherent)
            context.flags = RTC_INTERSECT_COHERENT;
        else
            context.flags = RTC_INTERSECT_INCOHERENT;
        context.userRayExt = 0;
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
        if (hint == rayHintCoherent)
            context.flags = RTC_INTERSECT_COHERENT;
        else
            context.flags = RTC_INTERSECT_INCOHERENT;
        context.userRayExt = 0;
        rtcOccludedNp(scene, &context, erays, count);

        for (int i = 0; i < count; i += simdSize)
            store(load(&geomIDs[i]) == vint(zero), &rays.far[i], vfloat(zero));
    }
};

} // namespace prt
