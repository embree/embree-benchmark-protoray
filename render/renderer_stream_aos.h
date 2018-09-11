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
#include "integrator.h"
#include "renderer.h"

namespace prt {

class RendererStreamAos
{
public:
    template <int streamSize>
    static Props queryRay(const ref<const Scene>& scene, const ref<IntersectorStreamAos<streamSize>>& intersector, const Ray& inputRay)
    {
        Props result;

        // Shoot the ray
        Ray ray = inputRay;
        RayHitStreamAos<streamSize> rays;
        rays.set(0, ray);
        ShadingContext ctx;
        RayStats stats;
        intersector->intersect(rays, 1, stats);
        rays.getRay(0, ray);
        Hit hit;
        rays.getHit(0, hit);
        if (none(ray.isHit())) return result;
        scene->postIntersect(ray, hit, ctx);
        int matId = scene->getMaterialId(hit.primId);

        // Fill the query result
        result.set("mat", scene->getMaterialName(matId));
        result.set("matId", matId);
        result.set("primId", hit.primId);
        result.set("dist", ray.far);
        result.set("p", ray.getHitPoint());
        result.set("Ng", ctx.Ng);
        result.set("N", ctx.f.N);
        result.set("uv", ctx.uv);
        result.set("eps", ctx.eps);

        return result;
    }
};

template <int streamSize>
FORCEINLINE int rayStreamSort(const RayHitStreamAos<streamSize>& ray, int* pathId, int rayCount)
{
    int missCount = 0;
    for (int i = 0; i < rayCount; ++i)
        if (!ray.isHit(i))
            missCount++;

    int missIndex = 0;
    int hitIndex = missCount;
    for (int i = 0; i < rayCount; ++i)
    {
        if (ray.isHit(i))
            pathId[hitIndex++] = i;
        else
            pathId[missIndex++] = i;
    }

    return missCount;
}

} // namespace prt
