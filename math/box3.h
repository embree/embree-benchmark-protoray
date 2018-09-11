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

#include "vec3.h"

namespace prt {

template <class T>
struct Box3
{
    Vec3<T> low;
    Vec3<T> high;

    FORCEINLINE Box3() {}
    FORCEINLINE Box3(Empty) : low(posMax), high(negMax) {}
    FORCEINLINE Box3(const Box3<T>& box) : low(box.low), high(box.high) {}
    FORCEINLINE Box3(const Vec3<T>& low, const Vec3<T>& high) : low(low), high(high) {}

    FORCEINLINE Box3<T>& operator =(const Box3<T>& box)
    {
        low = box.low;
        high = box.high;
        return *this;
    }

    FORCEINLINE Vec3<T>& operator [](size_t i)
	{
		assert(i < 2);
        return (&low)[i];
	}

    FORCEINLINE const Vec3<T>& operator [](size_t i) const
	{
		assert(i < 2);
        return (&low)[i];
	}

    FORCEINLINE Vec3<T> getSize() const
    {
        return high - low;
    }

    FORCEINLINE Vec3<T> getCenter() const
    {
        return (low + high) * 0.5f;
    }

    FORCEINLINE T getArea() const
    {
        return 2 * getHalfArea();
    }

    FORCEINLINE T getHalfArea() const
    {
        Vec3<T> s = getSize();
        return s.x * (s.y + s.z) + s.y * s.z;
    }

    FORCEINLINE T getVolume() const
    {
        Vec3<T> s = getSize();
        return s.x * s.y * s.z;
    }

    FORCEINLINE void grow(const Vec3<T>& vec)
	{
        low = min(low, vec);
        high = max(high, vec);
	}

    FORCEINLINE void grow(const Box3<T>& box)
	{
        low = min(low, box.low);
        high = max(high, box.high);
	}

    FORCEINLINE void intersect(const Box3<T>& box)
	{
        low = max(low, box.low);
        high = min(high, box.high);
	}

    FORCEINLINE bool contains(const Box3<T>& box) const
	{
		for (int i = 0; i < 3; ++i)
		{
            if (box.low[i] < low[i] || box.high[i] > high[i])
				return false;
		}

		return true;
	}

    FORCEINLINE bool overlaps(const Box3<T>& box) const
	{
		for (int i = 0; i < 3; ++i)
		{
            if (high[i] < box.low[i] || low[i] > box.high[i])
				return false;
		}

		return true;
	}

    FORCEINLINE bool isValid() const
	{
		for (int i = 0; i < 3; ++i)
		{
            if (low[i] > high[i])
				return false;
		}

		return true;
	}
};

typedef Box3<float> Box3f;
typedef Box3<double> Box3d;
typedef Box3<int> Box3i;

} // namespace prt
