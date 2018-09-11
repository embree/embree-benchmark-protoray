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

#include <ctime>
#include "sys/common.h"
#include "sys/tick_counter.h"
#include "vec2.h"
#include "vec3.h"

namespace prt {

class Random
{
private:
    int seed;
    int state;
    int table[32];

public:
    Random(int s = 27)
	{
        reset(s);
	}

	void reset(int s = 27)
    {
		const int a = 16807;
		const int m = 2147483647;
		const int q = 127773;
		const int r = 2836;
		int j, k;

		if (s == 0)
            seed = 1;
        else if (s < 0)
            seed = -s;
        else
            seed = s;

		for (j = 32+7; j >= 0; --j)
		{
            k = seed / q;
            seed = a * (seed - k * q) - r * k;

            if (seed < 0)
                seed += m;

			if (j < 32)
                table[j] = seed;
        }

        state = table[0];
    }

	int getInt()
    {
		const int a = 16807;
		const int m = 2147483647;
		const int q = 127773;
		const int r = 2836;

        int k = seed / q;
        seed = a * (seed - k * q) - r * k;
        if (seed < 0)
            seed += m;

        int j = state / (1 + (2147483647-1) / 32);
        state = table[j];
        table[j] = seed;

        return state;
    }

	uint32_t getUint()
	{
        return (uint32_t)getInt();
	}

	float getFloat()
	{
        return (getUint() & 0xffffff) / (float)(1 << 24);
	}

    float getFloat(float a, float b)
    {
        return a + getFloat() * (b-a);
    }

    Vec2f getFloat2()
	{
		return Vec2f(getFloat(), getFloat());
	}

    Vec3f getFloat3()
	{
		return Vec3f(getFloat(), getFloat(), getFloat());
	}

    int getInt(int a, int b)
    {
        return min(a + int(getFloat() * (b-a+1)), b);
    }
};

inline int generateRandomSeed()
{
	unsigned int s0 = time(0);
	uint64_t s12 = TickCounter::now();
    unsigned int s1 = (unsigned int)(s12);
    unsigned int s2 = (unsigned int)(s12 >> 32);

	return s0 ^ s1 ^ s2;
}

} // namespace prt
