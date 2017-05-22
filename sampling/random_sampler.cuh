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

#include "math/math.cuh"

namespace prt {

class RandomSampler
{
private:
    unsigned int state;

public:
    CUDA_DEV_FORCEINLINE void init(int pass, int pixelIndex)
    {
        unsigned int hash = 0;
        hash = murmurHash3Mix(hash, pixelIndex);
        hash = murmurHash3Mix(hash, pass);
        hash = murmurHash3Finalize(hash);

        state = hash;
    }

    CUDA_DEV_FORCEINLINE void init(int state)
    {
        this->state = state;
    }

    CUDA_DEV_FORCEINLINE float get1D()
    {
        state = lcgNext(state);
        return toFloatUnorm(state);
    }

    CUDA_DEV_FORCEINLINE float2 get2D()
    {
        state = lcgNext(state);
        float x = toFloatUnorm(state);

        state = lcgNext(state);
        float y = toFloatUnorm(state);

        return make_float2(x, y);
    }

    CUDA_DEV_FORCEINLINE unsigned int getState() const
    {
        return state;
    }

private:
    CUDA_DEV_FORCEINLINE unsigned int murmurHash3Mix(unsigned int hash, unsigned int k)
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

    CUDA_DEV_FORCEINLINE unsigned int murmurHash3Finalize(unsigned int hash)
    {
        hash ^= shr(hash, 16);
        hash *= 0x85ebca6b;
        hash ^= shr(hash, 13);
        hash *= 0xc2b2ae35;
        hash ^= shr(hash, 16);

        return hash;
    }

    CUDA_DEV_FORCEINLINE unsigned int lcgNext(unsigned int value)
    {
        const unsigned int m = 1664525;
        const unsigned int n = 1013904223;

        return value * m + n;
    }
};

} // namespace prt
