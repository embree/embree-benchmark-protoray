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

#include "sys/common.h"
#include "simd.h"
#include "vec2.h"
#include "vec3.h"

namespace prt {

class FastRandomSimd
{
private:
    vint value;

public:
    FORCEINLINE FastRandomSimd(uint32_t seed = 1)
    {
        reset(seed);
    }

    FORCEINLINE void reset(uint32_t seed = 1)
    {
        int x = seed;
        for (int i = 0; i < simdSize; ++i)
        {
            x = (x * 8191) ^ 140167;
            value[i] = x;
        }
    }

    FORCEINLINE vint getInt()
    {
        next();
        return value;
    }

    FORCEINLINE vfloat getFloat()
    {
        next();
        return toFloatUnorm(value);
    }

    FORCEINLINE Vec2vf getFloat2()
    {
        return Vec2vf(getFloat(), getFloat());
    }

    FORCEINLINE Vec3vf getFloat3()
    {
        return Vec3vf(getFloat(), getFloat(), getFloat());
    }

private:
    FORCEINLINE void next()
    {
        const vint multiplier = 1664525;
        const vint increment = 1013904223;
        value = multiplier * value + increment;
    }
};

} // namespace prt
