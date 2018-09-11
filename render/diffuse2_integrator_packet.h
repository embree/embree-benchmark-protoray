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

// Diffuse path tracing with next event estimation (shadow rays)
template <class ShadingContextT, class Sampler>
class Diffuse2IntegratorPacket : public IntegratorBase
{
private:
    int maxDepth;

public:
    Diffuse2IntegratorPacket(const Props& props)
    {
        maxDepth = props.get("maxDepth", 6);
    }

    int getSampleSize() const
    {
        return sampleDimBaseSize + 4 * maxDepth;
    }

    Vec3vf getColor(RaySimd& ray, IntersectorPacket* intersector, const Scene* scene, Sampler& sampler, IntegratorState<Sampler>& state)
    {
        static const vfloat R = 0.8f;

        vfloat L = zero;
        vfloat throughput = 1.f;

        HitSimd hit;
        vbool m = one;

        int depth = 0;
        for (; ;)
        {
            // Shoot the extension ray
            RayHint rayHint = (depth == 0) ? rayHintCoherent : rayHintIncoherent;
            intersector->intersect(m, ray, hit, state.rayStats, rayHint);

            vbool mMiss = andn(m, ray.isHit());
            if (any(mMiss))
            {
                set(mMiss, L, L + (throughput * ((depth > 0) ? 0.5f : 1.f))); // with MIS weight
                m = andn(m, mMiss);
                if (none(m))
                    break;
            }

            // Path termination
            if (depth == maxDepth)
                break;

            // Shade the hit point
            ShadingContextT ctx;
            scene->postIntersect(m, ray, hit, ctx);
            throughput *= R;

            // Generate and shoot a shadow ray
            Vec2vf s = sampler.get2D(state.sampler, sampleDimBaseSize + 4 * depth);
            ray.init(ctx.p, ctx.getFrame() * cosineSampleHemisphere(s), ctx.eps);

            intersector->occluded(m, ray, state.rayStats);

            vbool mDirMiss = andn(m, ray.isOccluded());
            if (any(mDirMiss))
                set(mDirMiss, L, L + (throughput * 0.5f)); // with MIS weight

            // Generate an extension ray
            s = sampler.get2D(state.sampler, sampleDimBaseSize + 4 * depth + 2);
            ray.init(ctx.p, ctx.getFrame() * cosineSampleHemisphere(s), ctx.eps);

            depth++;
        }

        return L;
    }
};

} // namespace prt
