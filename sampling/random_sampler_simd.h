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

#include "math/vec2.h"

namespace prt {

class RandomSampler;

class RandomSamplerSimd
{
public:
    typedef RandomSampler Scalar;

    struct State
    {
        vint s;
    };

    void init(int sampleSize, int sampleCount, int pixelCount)
    {
    }

    FORCEINLINE void setSample(vbool m, State& state, vint pass, vint pixelIndex)
    {
        vint hash = 0;
        hash = murmurHash3Mix(hash, pixelIndex);
        hash = murmurHash3Mix(hash, pass);
        hash = murmurHash3Finalize(hash);

        state.s = hash;
    }

    FORCEINLINE void resetSample(vbool m, State& state, vint pass, vint pixelIndex)
    {
    }

    FORCEINLINE vfloat get1D(State& state, vint dimension)
    {
        state.s = lcgNext(state.s);
        return toFloatUnorm(state.s);
    }

    FORCEINLINE Vec2vf get2D(State& state, vint dimension)
    {
        state.s = lcgNext(state.s);
        vfloat x = toFloatUnorm(state.s);

        state.s = lcgNext(state.s);
        vfloat y = toFloatUnorm(state.s);

        return Vec2vf(x, y);
    }

private:
    FORCEINLINE vint murmurHash3Mix(vint hash, vint k)
    {
        const unsigned int c1 = 0xcc9e2d51;
        const unsigned int c2 = 0x1b873593;
        const unsigned int r1 = 15;
        const unsigned int r2 = 13;
        const unsigned int m = 5;
        const unsigned int n = 0xe6546b64;

        k *= c1;
        k = shl(k, r1) | shr(k, 32 - r1);
        k *= c2;

        hash ^= k;
        hash = (shl(hash, r2) | shr(hash, 32 - r2)) * m + n;

        return hash;
    }

    FORCEINLINE vint murmurHash3Finalize(vint hash)
    {
        hash ^= shr(hash, 16);
        hash *= 0x85ebca6b;
        hash ^= shr(hash, 13);
        hash *= 0xc2b2ae35;
        hash ^= shr(hash, 16);

        return hash;
    }

    FORCEINLINE vint lcgNext(vint value)
    {
        const unsigned int m = 1664525;
        const unsigned int n = 1013904223;

        return value * m + n;
    }
};

} // namespace prt
