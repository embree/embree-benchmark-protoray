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
#include "integrator.h"
#include "renderer.h"

namespace prt {

class RendererStream
{
public:
    template <int streamSize>
    static Props queryPixel(const ref<IntersectorStream<streamSize>>& intersector, const Vec2i& imageSize, const Camera* camera, int x, int y)
    {
        Props result;

        CameraSampleSimd cameraSample;
        cameraSample.lens = zero;

        // Generate a ray through the center of the image plane
        // We need this to compute the depth
        RaySimd centerRay;
        cameraSample.image = Vec2f(0.5f);
        camera->getRay(centerRay, cameraSample);

        // Generate a ray through the pixel
        RaySimd ray;
        cameraSample.image = (Vec2f(x, y) + 0.5f) / toFloat(imageSize);
        camera->getRay(ray, cameraSample);

        // Shoot the ray
        RayStream<streamSize> rays;
        rays.setA(0, ray);
        HitStream<streamSize> hits;
        //ShadingContext ctx;
        RayStats stats;
        intersector->intersect(rays, hits, 1, stats);
        rays.getA(0, ray);
        if (none(ray.isHit())) return result;
        //scene->postIntersect(ray, hit, ctx);

        // Fill the query result
        //result.set("mat", ctx->scene->getMaterialName(ctx));
        //result.set("matId", ctx.matId);
        //result.set("prim", hit.id);
        result.set("depth", toScalar(ray.far * dot(ray.dir, centerRay.dir)));
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
FORCEINLINE int rayStreamSort(const RayStream<streamSize>& ray, int* pathId, int rayCount)
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

struct RayStreamMaterialIdSort
{
    static const int maxMatCount = 2048;
    ALIGNED_CACHE int buckets[maxMatCount];

    template <int streamSize>
    FORCEINLINE void operator ()(const Scene& scene, const RayStream<streamSize>& ray, const HitStream<streamSize>& hit, int* matId, int* pathId, int rayCount)
    {
        int matCount = scene.getMaterialCount();

        #pragma nounroll
        for (int i = 0; i < matCount; i += simdSize*2)
        {
            store(buckets + i, vint(zero));
            store(buckets + i + simdSize, vint(zero));
        }

        for (int i = 0; i < rayCount; ++i)
        {
            int id = ray.isHit(i) ? scene.getMaterialId(hit.primId[i]) : 0;
            matId[i] = id;
            buckets[id]++;
        }

        int sum = 0;
        for (int i = 0; i < matCount; ++i)
        {
            int bucket = buckets[i];
            buckets[i] = sum;
            sum += bucket;
        }
        for (int i = 0; i < rayCount; ++i)
        {
            int id = matId[i];
            pathId[buckets[id]++] = i;
        }
    }
};

} // namespace prt
