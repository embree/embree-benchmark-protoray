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

#include "sys/constants.h"

namespace prt {

template <class T>
struct Box1
{
    T low;
    T high;

    FORCEINLINE Box1() {}
    FORCEINLINE Box1(Empty) : low(posMax), high(negMax) {}
    FORCEINLINE Box1(const Box1<T>& box) : low(box.low), high(box.high) {}
    FORCEINLINE Box1(const T& low, const T& high) : low(low), high(high) {}

    FORCEINLINE Box1<T>& operator =(const Box1<T>& box)
    {
        low = box.low;
        high = box.high;
        return *this;
    }

    FORCEINLINE T& operator [](size_t i)
	{
		assert(i < 2);
        return (&low)[i];
	}

    FORCEINLINE const T& operator [](size_t i) const
	{
		assert(i < 2);
        return (&low)[i];
	}

    FORCEINLINE T getSize() const
    {
        return high - low;
    }

    FORCEINLINE T getCenter() const
    {
        return (low + high) * 0.5f;
    }

    FORCEINLINE T getArea() const
    {
        return high - low;
    }

    FORCEINLINE T getVolume() const
    {
        return high - low;
    }

    FORCEINLINE void grow(const T& x)
	{
        low = min(low, x);
        high = max(high, x);
	}

    FORCEINLINE void grow(const Box1<T>& box)
	{
        grow(box.low);
        grow(box.high);
	}

    FORCEINLINE void intersect(const Box1<T>& box)
	{
        low = max(low, box.low);
        high = min(high, box.high);
	}

    FORCEINLINE bool contains(const Box1& box) const
	{
        return box.low >= low && box.high <= high;
	}

    FORCEINLINE bool isValid() const
	{
        return low <= high;
	}
};

typedef Box1<float> Box1f;
typedef Box1<int> Box1i;

} // namespace prt
