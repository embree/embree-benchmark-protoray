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

        RTCRayHit eray;
        initRay(ray, eray.ray);
        initHit(eray.hit);
        rtcIntersect1(scene, &context, &eray);

        ray.far = eray.ray.tfar;
        hit.primId = eray.hit.primID;
        hit.u = eray.hit.u;
        hit.v = eray.hit.v;
    }

    void occluded(Ray& ray, RayStats& stats, RayHint hint)
    {
        RTCIntersectContext context;
        initIntersectContext(context, hint);

        stats.rayCount++;

        RTCRay eray;
        initRay(ray, eray);
        rtcOccluded1(scene, &context, &eray);

        if (eray.tfar < 0.f)
            ray.far = 0.f;
    }
};

} // namespace prt
