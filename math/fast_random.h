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

#include "sys/common.h"
#include "vec2.h"
#include "vec3.h"

namespace prt {

class FastRandom
{
private:
    uint32_t value;

public:
    FORCEINLINE FastRandom(uint32_t seed = 1) : value(seed) {}

    FORCEINLINE void reset(uint32_t seed = 1)
	{
        value = (seed * 8191) ^ 140167;
	}

    FORCEINLINE uint32_t getUint()
	{
		next();
        return value;
	}

    FORCEINLINE int getInt()
	{
		next();
        return value;
	}

    FORCEINLINE float getFloat()
	{
		next();
        return toFloatUnorm(value);
	}

    FORCEINLINE Vec2f getFloat2()
	{
		return Vec2f(getFloat(), getFloat());
	}

    FORCEINLINE Vec3f getFloat3()
	{
		return Vec3f(getFloat(), getFloat(), getFloat());
	}

private:
    FORCEINLINE void next()
	{
		const uint32_t multiplier = 1664525;
		const uint32_t increment = 1013904223;
        value = multiplier * value + increment;
	}
};

} // namespace prt
