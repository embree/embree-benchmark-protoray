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
#include "embree_intersector.h"

namespace prt {

class EmbreeIntersectorSingle : public IntersectorSingle, EmbreeIntersector
{
public:
    EmbreeIntersectorSingle(ref<const TriangleMesh> mesh, const Props& props, Props& stats) : EmbreeIntersector(mesh, props, stats) {}

    void intersect(Ray& ray, Hit& hit, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount++;

        RTCRay eray;
        initRay(ray, eray);
        rtcIntersect1Ex(scene, &context, eray);

        ray.far = eray.tfar;
        hit.primId = eray.primID;
        hit.u = eray.u;
        hit.v = eray.v;
    }

    void occluded(Ray& ray, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount++;

        RTCRay eray;
        initRay(ray, eray);
        rtcOccluded1Ex(scene, &context, eray);

        if (eray.geomID == 0)
            ray.far = 0.0f;
    }

private:
    FORCEINLINE void initRay(const Ray& ray, RTCRay& eray)
    {
        eray.org[0] = ray.org.x;
        eray.org[1] = ray.org.y;
        eray.org[2] = ray.org.z;

        eray.dir[0] = ray.dir.x;
        eray.dir[1] = ray.dir.y;
        eray.dir[2] = ray.dir.z;

        eray.tnear = 0.0f;
        eray.tfar = ray.far;

        eray.geomID = RTC_INVALID_GEOMETRY_ID;
        eray.primID = RTC_INVALID_GEOMETRY_ID;
        //eray.instID = RTC_INVALID_GEOMETRY_ID;

        eray.mask = -1;
        eray.time = 0.0f;
    }
};

} // namespace prt
