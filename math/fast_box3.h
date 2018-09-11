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

#include "box3.h"

namespace prt {

struct FastBox3f
{
    vfloat4 low;
    vfloat4 high;

    FORCEINLINE FastBox3f() {}
    FORCEINLINE FastBox3f(Empty) : low(posMax), high(negMax) {}
    FORCEINLINE FastBox3f(const FastBox3f& box) : low(box.low), high(box.high) {}
    FORCEINLINE FastBox3f(const vfloat4& low, const vfloat4& high) : low(low), high(high) {}
    FORCEINLINE explicit FastBox3f(const Box3f& box) : low(packVec3f(box.low)), high(packVec3f(box.high)) {}

    FORCEINLINE FastBox3f& operator =(const FastBox3f& box)
    {
        low = box.low;
        high = box.high;
        return *this;
    }

    FORCEINLINE vfloat4& operator [](size_t i)
	{
		assert(i < 2);
		return (&low)[i];
	}
	
    FORCEINLINE const vfloat4& operator [](size_t i) const
	{
		assert(i < 2);
		return (&low)[i];
	}

    FORCEINLINE vfloat4 getSize() const
    {
        return high - low;
    }

    FORCEINLINE vfloat4 getCenter() const
    {
        return (low + high) * 0.5f;
    }

    FORCEINLINE float getArea() const
    {
        return 2.0f * getHalfArea();
    }

    FORCEINLINE float getHalfArea() const
    {
        vfloat4 s = getSize();
        vfloat4 a = s * permute4<1,2,0,3>(s);
        return toScalar(a + broadcast<1>(a) + broadcast<2>(a));
    }

    FORCEINLINE float getVolume() const
    {
        vfloat4 s = getSize();
        vfloat4 v = s * permute4<1,2,0,3>(s) * permute4<2,0,1,3>(s);
        return toScalar(v);
    }

    FORCEINLINE Box3f unpack() const
    {
        return Box3f(unpackVec3f(low), unpackVec3f(high));
    }

    FORCEINLINE void grow(const vfloat4& vec)
	{
		low = min(low, vec);
		high = max(high, vec);
	}

    FORCEINLINE void grow(const FastBox3f& box)
	{
		low = min(low, box.low);
		high = max(high, box.high);
	}

    FORCEINLINE void intersect(const FastBox3f& box)
	{
		low = max(low, box.low);
		high = min(high, box.high);
	}
};

} // namespace prt
