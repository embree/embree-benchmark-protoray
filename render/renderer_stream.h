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
#include "integrator.h"
#include "renderer.h"

namespace prt {

class RendererStream
{
public:
    template <int streamSize>
    static Props queryRay(const ref<const Scene>& scene, const ref<IntersectorStream<streamSize>>& intersector, const Ray& inputRay)
    {
        Props result;

        // Shoot the ray
        RaySimd ray;
        ray.org = inputRay.org;
        ray.dir = inputRay.dir;
        ray.far = inputRay.far;
        RayStream<streamSize> rays;
        rays.setA(0, ray);
        HitStream<streamSize> hits;
        ShadingContextSimd ctx;
        RayStats stats;
        intersector->intersect(rays, hits, 1, stats);
        rays.getA(0, ray);
        HitSimd hit;
        hits.getA(0, hit);
        if (none(ray.isHit())) return result;
        scene->postIntersect(1, ray, hit, ctx);
        int primId = *hits.getPrimId();
        int matId = scene->getMaterialId(primId);

        // Fill the query result
        result.set("mat", scene->getMaterialName(matId));
        result.set("matId", matId);
        result.set("prim", primId);
        result.set("dist", toScalar(ray.far));
        result.set("p", toScalar(ray.getHitPoint()));
        result.set("Ng", toScalar(ctx.Ng));
        result.set("N", toScalar(ctx.f.N));
        result.set("uv", toScalar(ctx.uv));
        result.set("U", toScalar(ctx.f.U));
        result.set("V", toScalar(ctx.f.V));
        result.set("eps", toScalar(ctx.eps));

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
