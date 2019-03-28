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

#include "memory.h"

namespace prt {

// Raw array with uninitialized items
template <class T>
class RawArray : Uncopyable
{
    static_assert(is_trivially_destructible<T>::value, "data type must be POD");

private:
    T* items;

public:
    FORCEINLINE RawArray() : items(0) {}

    RawArray(int n)
    {
        assert(n >= 0);
        items = (T*)alignedAlloc(n * sizeof(T));
    }

    FORCEINLINE ~RawArray()
    {
        alignedFree(items);
    }

    FORCEINLINE T& operator [](size_t i)
    {
        return items[i];
    }

    FORCEINLINE const T& operator [](size_t i) const
    {
        return items[i];
    }

    // Reallocates the array deleting its previous contents
    void alloc(int n)
    {
        assert(n >= 0);
        alignedFree(items);
        items = (T*)alignedAlloc(n * sizeof(T));
    }

    void free()
    {
        alignedFree(items);
        items = 0;
    }

    FORCEINLINE T* getData()
    {
        return items;
    }

    FORCEINLINE const T* getData() const
    {
        return items;
    }
};

} // namespace prt
