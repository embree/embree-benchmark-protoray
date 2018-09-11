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

#include "sampling/shape_sampler.h"
#include "integrator.h"

namespace prt {

template <class ShadingContextT, class Sampler>
class AoIntegratorPacket : public IntegratorBase
{
private:
    int sampleCount;
    float invSampleCount;

public:
    AoIntegratorPacket(const Props& props)
    {
        sampleCount = props.get("samples", 16);
        invSampleCount = 1.0f / (float)sampleCount;
    }

    int getSampleSize() const
    {
        return sampleDimBaseSize + 2 * sampleCount;
    }

    Vec3vf getColor(RaySimd& ray, IntersectorPacket* intersector, const Scene* scene, Sampler& sampler, IntegratorState<Sampler>& state)
    {
        HitSimd hit;
        intersector->intersect(one, ray, hit, state.rayStats, rayHintCoherent);
        vbool active = ray.isHit();
        if (none(active)) return zero;

        ShadingContextT ctx;
        scene->postIntersect(active, ray, hit, ctx);

        vfloat sum = zero;
        for (int i = 0; i < sampleCount; ++i)
        {
            Vec2vf s = sampler.get2D(state.sampler, sampleDimBaseSize + 2 * i);
            ray.init(ctx.p, ctx.getFrame() * cosineSampleHemisphere(s), ctx.eps);

            //intersector->intersect(active, ray, hit, state.rayStats);
            intersector->occluded(active, ray, state.rayStats);
            sum += select(ray.isHit(), vfloat(zero), vfloat(one));
        }

        return select(active, sum * invSampleCount, zero);
    }
};

} // namespace prt
