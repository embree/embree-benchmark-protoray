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

#include "core/intersector_stream_aos.h"
#include "embree_intersector.h"

namespace prt {

template <int streamSize>
class EmbreeIntersectorStreamAos : public IntersectorStreamAos<streamSize>, EmbreeIntersector
{
public:
    EmbreeIntersectorStreamAos(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(RayHitStreamAos<streamSize>& rays, int count, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += count;

        rtcIntersect1M(scene, &context, rays.get(), count, sizeof(RTCRayHit));
    }

    void occluded(RayStreamAos<streamSize>& rays, int count, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += count;

        rtcOccluded1M(scene, &context, rays.get(), count, sizeof(RTCRay));
    }
};

template <int streamSize>
class EmbreeIntersectorSingleStreamAos : public IntersectorStreamAos<streamSize>, EmbreeIntersector
{
public:
    EmbreeIntersectorSingleStreamAos(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(RayHitStreamAos<streamSize>& rays, int count, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += count;

        for (int i = 0; i < count; ++i)
            rtcIntersect1(scene, &context, &rays[i]);
    }

    void occluded(RayStreamAos<streamSize>& rays, int count, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += count;

        for (int i = 0; i < count; ++i)
            rtcOccluded1(scene, &context, &rays[i]);
    }
};

// AOP intersector only for testing
template <int streamSize>
class EmbreeIntersectorStreamAop : public IntersectorStreamAos<streamSize>, EmbreeIntersector
{
public:
    EmbreeIntersectorStreamAop(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(RayHitStreamAos<streamSize>& rays, int count, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += count;

        RTCRayHit* rayPtrs[streamSize];
        for (int i = 0; i < count; ++i)
            rayPtrs[i] = &rays[i];

        rtcIntersect1Mp(scene, &context, rayPtrs, count);
    }

    void occluded(RayStreamAos<streamSize>& rays, int count, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount += count;

        RTCRay* rayPtrs[streamSize];
        for (int i = 0; i < count; ++i)
            rayPtrs[i] = &rays[i];

        rtcOccluded1Mp(scene, &context, rayPtrs, count);
    }
};

} // namespace prt
