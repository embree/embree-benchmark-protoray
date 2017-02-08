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

#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include "common.h"
#include "stream.h"

//#define DEBUG_MEMORY

#if defined(__AVX512F__) || defined(__MIC__)
#define SIMD_REG_SIZE 64 // AVX-512
#else
#define SIMD_REG_SIZE 32 // AVX
#endif

#define CACHE_LINE_SIZE 64

#if !defined(PAGE_SIZE)
#define PAGE_SIZE 4096
#endif

#ifdef _MSC_VER
#define ALIGNED(ALIGNMENT) __declspec(align(ALIGNMENT))
#else
#define ALIGNED(ALIGNMENT) __attribute__((aligned(ALIGNMENT)))
#endif

#define ALIGNED_SIMD ALIGNED(SIMD_REG_SIZE)
#define ALIGNED_CACHE ALIGNED(CACHE_LINE_SIZE)


namespace prt {

const size_t cacheLineSize = CACHE_LINE_SIZE;
const size_t pageSize = PAGE_SIZE;

void* alignedAlloc(size_t size, size_t alignment = 0);
void alignedFree(void* ptr);

// Typed copy
template <class T>
inline void copy(T* dest, const T* src, size_t count)
{
    memcpy(dest, src, count * sizeof(T));
}

// Prefetching
// -----------

FORCEINLINE void prefetchL1(const void* data, int offset = 0)
{
    _mm_prefetch((const char*)data+offset, _MM_HINT_T0);
}

FORCEINLINE void prefetchL2(const void* data, int offset = 0)
{
    _mm_prefetch((const char*)data+offset, _MM_HINT_T1);
}

FORCEINLINE void prefetchL1Ex(const void* data, int offset = 0)
{
#ifdef __MIC__
    _mm_prefetch((const char*)data+offset, _MM_HINT_ET0);
#else
    _mm_prefetch((const char*)data+offset, _MM_HINT_T0);
#endif
}

FORCEINLINE void prefetchL2Ex(const void* data, int offset = 0)
{
#ifdef __MIC__
    _mm_prefetch((const char*)data+offset, _MM_HINT_ET1);
#else
    _mm_prefetch((const char*)data+offset, _MM_HINT_T1);
#endif
}

FORCEINLINE void prefetch2L1(const void* data, int offset = 0)
{
    _mm_prefetch((const char*)data+offset,    _MM_HINT_T0);
    _mm_prefetch((const char*)data+offset+64, _MM_HINT_T0);
}

FORCEINLINE void prefetch2L1Ex(const void* data, int offset = 0)
{
#ifdef __MIC__
    _mm_prefetch((const char*)data+offset,    _MM_HINT_ET0);
    _mm_prefetch((const char*)data+offset+64, _MM_HINT_ET0);
#else
    _mm_prefetch((const char*)data+offset,    _MM_HINT_T0);
    _mm_prefetch((const char*)data+offset+64, _MM_HINT_T0);
#endif
}

FORCEINLINE void prefetch2L2Ex(const void* data, int offset = 0)
{
#ifdef __MIC__
    _mm_prefetch((const char*)data+offset,    _MM_HINT_ET1);
    _mm_prefetch((const char*)data+offset+64, _MM_HINT_ET1);
#else
    _mm_prefetch((const char*)data+offset,    _MM_HINT_T1);
    _mm_prefetch((const char*)data+offset+64, _MM_HINT_T1);
#endif
}

// Memory object
// -------------

class Memory
{
private:
    void* data;
    size_t size;

public:
    FORCEINLINE Memory() : data(0), size(0) {}

    Memory(size_t size)
    {
        this->data = alignedAlloc(size);
        this->size = size;
    }

    Memory(const Memory& other)
    {
        if (other.data)
        {
            data = alignedAlloc(other.size);
            size = other.size;
            memcpy(data, other.data, size);
        }
        else
        {
            data = 0;
            size = 0;
        }
    }

    FORCEINLINE ~Memory()
    {
        if (data) free();
    }

    Memory& operator =(const Memory& other)
    {
        if (this != &other)
        {
            if (data) free();

            if (other.data)
            {
                data = alignedAlloc(other.size);
                size = other.size;
                memcpy(data, other.data, size);
            }
        }

        return *this;
    }

    void alloc(size_t size)
    {
        if (data) free();

        data = alignedAlloc(size);
        this->size = size;
    }

    void free()
    {
        alignedFree(data);
        data = 0;
        size = 0;
    }

    FORCEINLINE void* getData()
    {
        return data;
    }

    FORCEINLINE const void* getData() const
    {
        return data;
    }

    FORCEINLINE size_t getSize() const
    {
        return size;
    }
};

Stream& operator >>(Stream& ism, Memory& mem);
Stream& operator <<(Stream& osm, const Memory& mem);

} // namespace prt
