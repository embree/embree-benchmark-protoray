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

#include "math/vec2.h"
#include "math/hash.h"

namespace prt {

class RandomSamplerSimd;

class RandomSampler
{
private:
    int seed = 0;

public:
    typedef RandomSamplerSimd Simd;

    struct State
    {
        unsigned int s;
    };

    void init(int dimensionCount, int sampleCount, int pixelCount, int seed = 0)
    {
        this->seed = seed;
    }

    FORCEINLINE void setSample(State& state, int pass, int pixelIndex)
    {
        int hash = seed;
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
};

} // namespace prt
