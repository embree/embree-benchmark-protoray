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

#include "integrator.h"

namespace prt {

template <class ShadingContextT, class Sampler>
class PrimaryIntegratorPacket : public IntegratorBase
{
public:
    PrimaryIntegratorPacket(const Props& props) {}

    int getSampleSize() const
    {
        return sampleDimBaseSize;
    }

    Vec3vf getColor(RaySimd& ray, IntersectorPacket* intersector, const Scene* scene, Sampler& sampler, IntegratorState<Sampler>& state)
    {
        HitSimd hit;
        vbool active = one;
        intersector->intersect(active, ray, hit, state.rayStats, rayHintCoherent);
        active = ray.isHit();

        Vec3vf color;
        if (any(active))
        {
            ShadingContextT ctx;
            scene->postIntersect(active, ray, hit, ctx);
            color = (ctx.getN() + vfloat(1.0f)) * vfloat(0.5f);
        }

        return select(active, color, Vec3vf(0.05f));
    }
};

} // namespace prt
