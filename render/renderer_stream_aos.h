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

#include "core/intersector_stream_aos.h"
#include "integrator.h"
#include "renderer.h"

namespace prt {

class RendererStreamAos
{
public:
    template <int streamSize>
    static Props queryPixel(const ref<IntersectorStreamAos<streamSize>>& intersector, const Vec2i& imageSize, const Camera* camera, int x, int y)
    {
        Props result;

        CameraSample cameraSample;
        cameraSample.lens = zero;

        // Generate a ray through the center of the image plane
        // We need this to compute the depth
        Ray centerRay;
        cameraSample.image = Vec2f(0.5f);
        camera->getRay(centerRay, cameraSample);

        // Generate a ray through the pixel
        Ray ray;
        cameraSample.image = (Vec2f(x, y) + 0.5f) / toFloat(imageSize);
        camera->getRay(ray, cameraSample);

        // Shoot the ray
        RayHitStreamAos<streamSize> rays;
        rays.set(0, ray);
        //ShadingContext ctx;
        RayStats stats;
        intersector->intersect(rays, 1, stats);
        rays.getRay(0, ray);
        if (none(ray.isHit())) return result;
        //scene->postIntersect(ray, hit, ctx);

        // Fill the query result
        //result.set("mat", ctx->scene->getMaterialName(ctx));
        //result.set("matId", ctx.matId);
        //result.set("prim", hit.id);
        result.set("depth", ray.far * dot(ray.dir, centerRay.dir));
        //result.set("p", ray.getHitPoint());
        //result.set("Ng", ctx.Ng);
        //result.set("N", ctx.N);
        //result.set("uv", ctx.uv);
        //result.set("U", ctx.U);
        //result.set("V", ctx.V);

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
