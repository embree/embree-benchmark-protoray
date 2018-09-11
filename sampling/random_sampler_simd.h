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

class RandomSampler;

class RandomSamplerSimd
{
private:
    int seed = 0;

public:
    typedef RandomSampler Scalar;

    struct State
    {
        vint s;
    };

    void init(int dimensionCount, int sampleCount, int pixelCount, int seed = 0)
    {
        this->seed = seed;
    }

    FORCEINLINE void setSample(vbool m, State& state, vint pass, vint pixelIndex)
    {
        vint hash = seed;
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
};

} // namespace prt
