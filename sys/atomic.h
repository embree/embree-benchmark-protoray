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

#include "common.h"
#include "memory.h"

namespace prt {

FORCEINLINE int atomicAdd(volatile int* ptr, int val)
{
    return __sync_fetch_and_add(ptr, val);
}

FORCEINLINE uint32_t atomicAdd(volatile uint32_t* ptr, uint32_t val)
{
    return __sync_fetch_and_add(ptr, val);
}

FORCEINLINE int atomicSwap(volatile int* ptr, int val)
{
    return __sync_lock_test_and_set(ptr, val);
}

FORCEINLINE uint32_t atomicSwap(volatile uint32_t* ptr, uint32_t val)
{
    return __sync_lock_test_and_set(ptr, val);
}

FORCEINLINE int atomicCas(volatile int* ptr, int oldVal, int newVal)
{
    return __sync_val_compare_and_swap(ptr, oldVal, newVal);
}

FORCEINLINE uint32_t atomicCas(volatile uint32_t* ptr, uint32_t oldVal, uint32_t newVal)
{
    return __sync_val_compare_and_swap(ptr, oldVal, newVal);
}

// Atomic counter
template <class T>
class Atomic
{
private:
    volatile T data;

public:
    FORCEINLINE Atomic() {}
    FORCEINLINE Atomic(T a) : data(a) {}
    FORCEINLINE Atomic(const Atomic<T>& a) : data(a.data) {}

    FORCEINLINE Atomic<T>& operator =(T a)
    {
        data = a;
        return *this;
    }

    FORCEINLINE Atomic<T>& operator =(const Atomic<T>& a)
    {
        data = a.data;
        return *this;
    }

    FORCEINLINE operator T() const { return data; }

    FORCEINLINE T operator +=(T a) { return atomicAdd(&data, a) + a; }
    FORCEINLINE T operator -=(T a) { return atomicAdd(&data, T(-a)) - a; }

    FORCEINLINE T operator ++() { return atomicAdd(&data, 1) + 1; }
    FORCEINLINE T operator --() { return atomicAdd(&data, T(-1)) - 1; }

    FORCEINLINE T operator ++(int) { return atomicAdd(&data, 1); }
    FORCEINLINE T operator --(int) { return atomicAdd(&data, T(-1)); }
};

// Aligned atomic counter
template <class T>
class ALIGNED_CACHE AlignedAtomic
{
private:
    volatile T data;

public:
    FORCEINLINE AlignedAtomic() {}
    FORCEINLINE AlignedAtomic(T a) : data(a) {}
    FORCEINLINE AlignedAtomic(const AlignedAtomic<T>& a) : data(a.data) {}

    FORCEINLINE AlignedAtomic<T>& operator =(T a)
    {
        data = a;
        return *this;
    }

    FORCEINLINE AlignedAtomic<T>& operator =(const AlignedAtomic<T>& a)
    {
        data = a.data;
        return *this;
    }

    FORCEINLINE operator T() const { return data; }

    FORCEINLINE T operator +=(T a) { return atomicAdd(&data, a) + a; }
    FORCEINLINE T operator -=(T a) { return atomicAdd(&data, T(-a)) - a; }

    FORCEINLINE T operator ++() { return atomicAdd(&data, 1) + 1; }
    FORCEINLINE T operator --() { return atomicAdd(&data, T(-1)) - 1; }

    FORCEINLINE T operator ++(int) { return atomicAdd(&data, 1); }
    FORCEINLINE T operator --(int) { return atomicAdd(&data, T(-1)); }
};

} // namespace prt
