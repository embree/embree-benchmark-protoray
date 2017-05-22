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

#include <algorithm>
#include <iostream>
#include "common.h"

namespace prt {

// Statically allocated array for POD types only!
template <class T, int capacity>
class StaticArray
{
private:
    T items[capacity];
    int size;

public:
    FORCEINLINE StaticArray() : size(0) {}

    FORCEINLINE StaticArray(int n)
	{
        assert(n >= 0 && n <= capacity);
        size = n;
	}

	FORCEINLINE T& operator [](size_t i)
	{
        assert(i < (size_t)size);
        return items[i];
	}

	FORCEINLINE const T& operator [](size_t i) const
	{
        assert(i < (size_t)size);
        return items[i];
	}

    FORCEINLINE T* getData()
	{
        return items;
	}

    FORCEINLINE const T* getData() const
	{
        return items;
	}

    FORCEINLINE int getSize() const
	{
        return size;
	}

	FORCEINLINE bool isEmpty() const
	{
        return size == 0;
	}

    void resize(int n)
	{
        assert(n >= 0 && n <= capacity);
        size = n;
	}

	void clear()
	{
        size = 0;
	}

	FORCEINLINE int pushBack()
	{
        assert(size < capacity);
        return size++;
	}

	FORCEINLINE int pushBack(const T& item)
	{
        assert(size < capacity);
        items[size] = item;
        return size++;
	}

    template <class... Items>
    int pushBack(const T& item, const Items&... moreItems)
    {
        pushBack(item);
        return pushBack(moreItems...);
    }

	// Return -1 if the item is not found
	int indexOf(const T& item)
	{
        for (int i = 0; i < size; ++i)
		{
            if (items[i] == item)
				return i;
		}

		return -1;
	}

	// Slow!
    void insert(int pos, const T& value)
	{
        assert(size < capacity);
        assert(pos >= 0 && pos <= size);

        memmove(items + pos + 1, items + pos, (size - pos) * sizeof(T));
        items[pos] = value;
        ++size;
	}

	// Slow!
    void remove(int pos)
	{
        assert(pos >= 0 && pos < size);

        memmove(items + pos, items + pos + 1, (size - pos - 1) * sizeof(T));
        --size;
	}

    void sort()
    {
        std::sort(items, items + size);
    }

    template <class Predicate>
    void sort(Predicate predicate)
    {
        std::sort(items, items + size, predicate);
    }

    FORCEINLINE T& getFront()
	{
        assert(size > 0);
        return items[0];
	}

    FORCEINLINE const T& getFront() const
	{
        assert(size > 0);
        return items[0];
	}

    FORCEINLINE T& getBack()
	{
        assert(size > 0);
        return items[size - 1];
	}

    FORCEINLINE const T& getBack() const
	{
        assert(size > 0);
        return items[size - 1];
	}

	FORCEINLINE T* begin()
	{
        return items;
	}

	FORCEINLINE const T* begin() const
	{
        return items;
	}

	FORCEINLINE T* end()
	{
        return items + size;
	}

	FORCEINLINE const T* end() const
	{
        return items + size;
	}
};

template <class T, int capacity>
inline std::ostream& operator <<(std::ostream& osm, const StaticArray<T, capacity>& array)
{
    osm << "{" << std::endl;
    for (int i = 0; i < array.getSize(); ++i)
	{
        osm << "    " << array[i] << std::endl;
	}
    osm << "}";
    return osm;
}

} // namespace prt
