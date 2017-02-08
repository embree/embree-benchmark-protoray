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

#include "sampling/shape_sampler.h"
#include "integrator.h"

namespace prt {

template <class Sampler>
class AoIntegratorSingle : public IntegratorBase
{
private:
    int sampleCount;
    float invSampleCount;

public:
    AoIntegratorSingle(const Props& props)
    {
        sampleCount = props.get("samples", 16);
        invSampleCount = 1.0f / (float)sampleCount;
    }

    int getSampleSize() const
    {
        return sampleDimBaseSize + 2 * sampleCount;
    }

    Vec3f getColor(Ray& ray, IntersectorSingle* intersector, const Scene* scene, Sampler& sampler, IntegratorState<Sampler>& state)
    {
        Hit hit;
        intersector->intersect(ray, hit, state.rayStats);
        if (!ray.isHit()) return zero;

        ShadingContext ctx;
        scene->postIntersect(ray, hit, ctx);

        float sum = zero;
        for (int i = 0; i < sampleCount; ++i)
        {
            Vec2f s = sampler.get2D(state.sampler, sampleDimBaseSize + 2 * i);
            ray.init(ctx.p, ctx.getBasis() * cosineSampleHemisphere(s), ctx.eps);

            intersector->occluded(ray, state.rayStats);
            if (!ray.isHit()) sum += 1.0f;
        }

        return sum * invSampleCount;
    }
};

} // namespace prt
