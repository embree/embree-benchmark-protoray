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

#include <stdexcept>
#include "color.h"
#include "tone_mapper.h"

namespace prt {

class ReinhardToneMapper : public ToneMapper
{
private:
    float exposure;
    float invWhite2;

public:
    ReinhardToneMapper(const Props& props)
    {
        exposure = props.get("exposure", 1.0f);
        float burn = props.get("burn", 0.0f);
        if (burn < 0.0f || burn > 1.0f)
            throw std::invalid_argument("burn must be between 0 and 1");

        //float white = 1.0f / burn;
        //invWhite2 = 1.0f / sqr(white);
        invWhite2 = sqr(burn);
    }

    Vec3f get(const Vec3f& c) const
    {
        Vec3f d = c * exposure;
        float y = luminance(d);

        float yMapped = y * (1.0f + y * invWhite2) * rcp(1.0f + y);
        return d * (yMapped * rcpSafe(y));
    }

    Vec3vf get(const Vec3vf& c) const
    {
        Vec3vf d = c * vfloat(exposure);
        vfloat y = luminance(d);

        vfloat yMapped = y * (1.0f + y * invWhite2) * rcp(1.0f + y);
        return d * (yMapped * rcpSafe(y));
    }
};

} // namespace prt
