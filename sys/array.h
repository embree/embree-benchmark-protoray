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

#include <type_traits>
#include <algorithm>
#include <iostream>
#include "memory.h"
#include "stream.h"

namespace prt {

// Aligned dynamic array
template <class T>
class Array
{
private:
    T* items;
    int size;
    int capacity;

public:
    FORCEINLINE Array() : items(0), size(0), capacity(0) {}

    Array(int n)
	{
        assert(n >= 0);

        items = (T*)alignedAlloc(n * sizeof(T));

        for (int i = 0; i < n; ++i)
            new (items + i) T;

        size = n;
        capacity = n;
	}

	Array(const Array& other)
	{
        copyFrom(other);
	}

    FORCEINLINE ~Array()
	{
		cleanup();
	}

	Array& operator =(const Array& other)
	{
		if (&other == this)
			return *this;
		
		cleanup();
        copyFrom(other);
		return *this;
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

	// Reallocates the array deleting its previous contents
    void alloc(int n)
	{
        assert(n >= 0);

		cleanup();

        items = (T*)alignedAlloc(n * sizeof(T));

        for (int i = 0; i < n; ++i)
            new (items + i) T;

        size = n;
        capacity = n;
	}

	void free()
	{
		cleanup();

        items = 0;
        size = 0;
        capacity = 0;
	}

    FORCEINLINE int getSize() const
	{
        return size;
	}

    FORCEINLINE bool isEmpty() const
	{
        return size == 0;
	}

	// Resizes the array (does not compact!)
    void resize(int n)
	{
        if (size == n)
			return;

        if (n < size)
		{
			// Trim
            for (int i = n; i < size; ++i)
                items[i].~T();

            size = n;
		}
		else
		{
			// Extend
            reserve(n);

            for (int i = size; i < n; ++i)
                new (items + i) T;

            size = n;
		}
	}

	// Reallocates the array
    void reserve(int n)
	{
        assert(n >= size);

        if (n == capacity)
			return;

        capacity = n;

        // Reallocate and move data
        T* oldItems = items;
        items = (T*)alignedAlloc(capacity * sizeof(T));

        if (std::is_trivially_copy_constructible<T>::value)
        {
            memcpy(items, oldItems, size * sizeof(T));
        }
        else
        {
            // Slow!
            for (int i = 0; i < size; ++i)
            {
                new (items + i) T(std::move(oldItems[i]));
                oldItems[i].~T();
            }
        }

        alignedFree(oldItems);
	}

    void compact()
	{
        reserve(size);
	}

    void clear()
	{
        for (int i = 0; i < size; ++i)
            items[i].~T();

        size = 0;
	}

    FORCEINLINE int pushBack()
	{
        if (size == capacity)
            expand();

        new (items + size) T;
        return size++;
	}

    FORCEINLINE int pushBack(const T& item)
	{
        if (size == capacity)
            expand();

        new (items + size) T(item);
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
        assert(pos >= 0 && pos <= size);

        if (size == capacity)
            expand();

        memmove(items + pos + 1, items + pos, (size - pos) * sizeof(T));
        new (&items[pos]) T(value);
        ++size;
	}

	// Slow!
    void remove(int pos)
	{
        assert(pos >= 0 && pos < size);

        items[pos].~T();
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

    FORCEINLINE T* getData()
	{
        return items;
	}

    FORCEINLINE const T* getData() const
	{
        return items;
	}

private:
	void cleanup()
	{
        for (int i = 0; i < size; ++i)
            items[i].~T();

        alignedFree(items);
	}

    void copyFrom(const Array& other)
    {
        size = other.size;
        capacity = other.size;

        if (size > 0)
        {
            items = (T*)alignedAlloc(size * sizeof(T));

            if (std::is_trivially_copy_constructible<T>::value)
            {
                memcpy(items, other.items, size * sizeof(T));
            }
            else
            {
                // Slow!
                for (int i = 0; i < size; ++i)
                    new (items + i) T(other.items[i]);
            }
        }
        else
        {
            items = 0;
        }
    }

    void expand()
	{
		const int minCapacity = 32;
        int newCapacity = max(capacity * 2, minCapacity);
        reserve(newCapacity);
	}
};

template <class T>
inline Stream& operator <<(Stream& osm, const Array<T>& array)
{
    osm << array.getSize();

    if (std::is_trivially_copy_constructible<T>::value)
    {
        osm.write(array.getData(), array.getSize() * sizeof(T));
    }
    else
    {
        for (int i = 0; i < array.getSize(); ++i)
            osm << array[i];
    }

    return osm;
}

template <class T>
inline Stream& operator >>(Stream& ism, Array<T>& array)
{
    int size;
    ism >> size;
    array.alloc(size);

    if (std::is_trivially_copy_constructible<T>::value)
    {
        ism.readFull(array.getData(), size * sizeof(T));
    }
    else
    {
        for (int i = 0; i < size; ++i)
            ism >> array[i];
    }

    return ism;
}

template <class T>
inline std::ostream& operator <<(std::ostream& osm, const Array<T>& array)
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
