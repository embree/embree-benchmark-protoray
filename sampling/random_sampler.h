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

class RandomSamplerSimd;

class RandomSampler
{
public:
    typedef RandomSamplerSimd Simd;

    struct State
    {
        unsigned int s;
    };

    void init(int sampleSize, int sampleCount, int pixelCount)
    {
    }

    FORCEINLINE void setSample(State& state, int pass, int pixelIndex)
    {
        unsigned int hash = 0;
        hash = murmurHash3Mix(hash, pixelIndex);
        hash = murmurHash3Mix(hash, pass);
        hash = murmurHash3Finalize(hash);

        state.s = hash;
    }

    FORCEINLINE void resetSample(State& state, int pass, int pixelIndex)
    {
    }

    FORCEINLINE float get1D(State& state, int dimension)
    {
        state.s = lcgNext(state.s);
        return toFloatUnorm(state.s);
    }

    FORCEINLINE Vec2f get2D(State& state, int dimension)
    {
        state.s = lcgNext(state.s);
        float x = toFloatUnorm(state.s);

        state.s = lcgNext(state.s);
        float y = toFloatUnorm(state.s);

        return Vec2f(x, y);
    }

private:
    FORCEINLINE unsigned int murmurHash3Mix(unsigned int hash, unsigned int k)
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

    FORCEINLINE unsigned int murmurHash3Finalize(unsigned int hash)
    {
        hash ^= shr(hash, 16);
        hash *= 0x85ebca6b;
        hash ^= shr(hash, 13);
        hash *= 0xc2b2ae35;
        hash ^= shr(hash, 16);

        return hash;
    }

    FORCEINLINE unsigned int lcgNext(unsigned int value)
    {
        const unsigned int m = 1664525;
        const unsigned int n = 1013904223;

        return value * m + n;
    }
};

} // namespace prt
