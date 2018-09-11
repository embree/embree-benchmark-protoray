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
class DiffuseIntegratorSingle : public IntegratorBase
{
private:
    int maxDepth;

public:
    DiffuseIntegratorSingle(const Props& props)
    {
        maxDepth = props.get("maxDepth", 6);
    }

    int getSampleSize() const
    {
        return sampleDimBaseSize + 2 * maxDepth;
    }

    Vec3f getColor(Ray& ray, IntersectorSingle* intersector, const Scene* scene, Sampler& sampler, IntegratorState<Sampler>& state)
    {
        const float R = 0.8f;
        float Lw = 1.0f;

        Hit hit;

        int depth = 0;
        for (; ;)
        {
            RayHint rayHint = (depth == 0) ? rayHintCoherent : rayHintIncoherent;
            intersector->intersect(ray, hit, state.rayStats, rayHint);

            if (!ray.isHit() || depth == maxDepth)
                break;

            Lw *= R;

            ShadingContextT ctx;
            scene->postIntersect(ray, hit, ctx);

            Vec2f s = sampler.get2D(state.sampler, sampleDimBaseSize + 2 * depth);
            ray.init(ctx.p, ctx.getFrame() * cosineSampleHemisphere(s), ctx.eps);
            depth++;
        }

        if (ray.isHit()) return zero;
        return Lw;
    }
};

} // namespace prt
