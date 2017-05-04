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

template <class ShadingContextT, class Sampler>
class DiffuseIntegratorPacket : public IntegratorBase
{
private:
    int maxDepth;

public:
    DiffuseIntegratorPacket(const Props& props)
    {
        maxDepth = props.get("maxDepth", 6);
    }

    int getSampleSize() const
    {
        return sampleDimBaseSize + 2 * maxDepth;
    }

    Vec3vf getColor(RaySimd& ray, IntersectorPacket* intersector, const Scene* scene, Sampler& sampler, IntegratorState<Sampler>& state)
    {
        static const vfloat R = 0.8f;
        vfloat Lw = 1.0f;

        HitSimd hit;
        vbool active = one;

        int depth = 0;
        for (; ;)
        {
            intersector->intersect(active, ray, hit, state.rayStats);
            active &= ray.isHit();
            if (none(active) || depth == maxDepth)
                break;

            set(active, Lw, Lw * R);

            ShadingContextT ctx;
            scene->postIntersect(active, ray, hit, ctx);

            Vec2vf s = sampler.get2D(state.sampler, sampleDimBaseSize + 2 * depth);
            ray.init(ctx.p, ctx.getBasis() * cosineSampleHemisphere(s), ctx.eps);
            depth++;
        }

        set(active, Lw, vfloat(zero));
        return Lw;
    }
};

} // namespace prt
