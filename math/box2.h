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

#include "vec2.h"

namespace prt {

template <class T>
struct Box2
{
    Vec2<T> low;
    Vec2<T> high;

    FORCEINLINE Box2() {}
    FORCEINLINE Box2(Empty) : low(posMax), high(negMax) {}
    FORCEINLINE Box2(const Box2<T>& box) : low(box.low), high(box.high) {}
    FORCEINLINE Box2(const Vec2<T>& low, const Vec2<T>& high) : low(low), high(high) {}

    FORCEINLINE Box2<T>& operator =(const Box2<T>& box)
    {
        low = box.low;
        high = box.high;
        return *this;
    }

    FORCEINLINE Vec2<T>& operator [](size_t i)
	{
		assert(i < 2);
        return (&low)[i];
	}

    FORCEINLINE const Vec2<T>& operator [](size_t i) const
	{
		assert(i < 2);
        return (&low)[i];
	}

    FORCEINLINE Vec2<T> getSize() const
    {
        return high - low;
    }

    FORCEINLINE Vec2<T> getCenter() const
    {
        return (low + high) * 0.5f;
    }

    FORCEINLINE T getArea() const
    {
        return 2 * getHalfArea();
    }

    FORCEINLINE T getHalfArea() const
    {
        Vec2<T> s = getSize();
        return s.x + s.y;
    }

    FORCEINLINE T getVolume() const
    {
        Vec2<T> s = getSize();
        return s.x * s.y;
    }

    FORCEINLINE void grow(const Vec2<T>& vec)
	{
        low = min(low, vec);
        high = max(high, vec);
	}

    FORCEINLINE void grow(const Box2<T>& box)
	{
        low = min(low, box.low);
        high = max(high, box.high);
	}

    FORCEINLINE void intersect(const Box2<T>& box)
	{
        low = max(low, box.low);
        high = min(high, box.high);
	}

    FORCEINLINE bool contains(const Box2<T>& box) const
	{
		for (int i = 0; i < 2; ++i)
		{
            if (box.low[i] < low[i] || box.high[i] > high[i])
				return false;
		}

		return true;
	}

    FORCEINLINE bool isValid() const
	{
		for (int i = 0; i < 2; ++i)
		{
            if (low[i] > high[i])
				return false;
		}

		return true;
	}
};

typedef Box2<float> Box2f;
typedef Box2<double> Box2d;
typedef Box2<int> Box2i;

} // namespace prt
