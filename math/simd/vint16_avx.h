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

#include <immintrin.h>
#include "../simd_common.h"
#include "vbool16_avx.h"

namespace prt {

// vint16
template <>
struct var<int,16>
{
    union
    {
        __m128i m[4];
        int v[16];
    };

    FORCEINLINE var() {}

    FORCEINLINE var(const vint16& a)
    {
        m[0] = a.m[0];
        m[1] = a.m[1];
        m[2] = a.m[2];
        m[3] = a.m[3];
    }

    FORCEINLINE var(__m128i a)
    {
        init(a);
    }

    FORCEINLINE var(int a)
    {
        init(a);
    }

    FORCEINLINE var(int a0, int a1, int a2, int a3)
    {
        init(a0, a1, a2, a3);
    }

    FORCEINLINE var(int a0, int a1, int a2,  int a3,  int a4,  int a5,  int a6,  int a7,
                   int a8, int a9, int a10, int a11, int a12, int a13, int a14, int a15)
    {
        init(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15);
    }

    FORCEINLINE var(Zero) { init(_mm_setzero_si128()); }
    FORCEINLINE var(One)  { init(1); }
    FORCEINLINE var(Step) { init(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); }

    FORCEINLINE vint16& operator =(const vint16& a)
    {
        m[0] = a.m[0];
        m[1] = a.m[1];
        m[2] = a.m[2];
        m[3] = a.m[3];
        return *this;
    }

    FORCEINLINE const int& operator [](size_t i) const
	{
		assert(i < 16);
        return v[i];
	}

    FORCEINLINE int& operator [](size_t i)
	{
		assert(i < 16);
        return v[i];
	}

private:
    // Helper functions
    // ----------------

    FORCEINLINE void init(__m128i a)
    {
        for (int i = 0; i < 4; ++i)
            m[i] = a;
    }

    FORCEINLINE void init(int a)
    {
        init(_mm_set1_epi32(a));
    }

    FORCEINLINE void init(int a0, int a1, int a2, int a3)
    {
        init(_mm_set_epi32(a3, a2, a1, a0));
    }

    FORCEINLINE void init(int a0, int a1, int a2,  int a3,  int a4,  int a5,  int a6,  int a7,
                         int a8, int a9, int a10, int a11, int a12, int a13, int a14, int a15)
    {
        m[0] = _mm_set_epi32(a3, a2, a1, a0);
        m[1] = _mm_set_epi32(a7, a6, a5, a4);
        m[2] = _mm_set_epi32(a11, a10, a9, a8);
        m[3] = _mm_set_epi32(a15, a14, a13, a12);
    }
};

// Load functions
// --------------

template <>
FORCEINLINE vint16 load(const int* ptr)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_load_si128((const __m128i*)(ptr + i * 4));
    return r;
}

template <>
FORCEINLINE vint16 load(const vbool16& mask, const int* ptr)
{
    vint16 r;
    for (int i = 0; i < 16; ++i)
        if (mask.m & (1 << i)) r[i] = ptr[i];
    return r;
}

FORCEINLINE vint16 uexpand(const vbool16& mask, const int* ptr)
{
    vint16 r;
    int offset = 0;
    for (int i = 0; i < 16; ++i)
        if (mask.m & (1 << i)) r[i] = ptr[offset++];
    return r;
}

FORCEINLINE vint16 gather(const int* ptr, const vint16& idx)
{
    vint16 r;
    for (int i = 0; i < 16; ++i)
        r[i] = ptr[idx[i]];
    return r;
}

FORCEINLINE vint16 gather(const vbool16& mask, const int* ptr, const vint16& idx)
{
    vint16 r;
    for (int i = 0; i < 16; ++i)
        if (mask.m & (1 << i)) r[i] = ptr[idx[i]];
    return r;
}

FORCEINLINE void prefetchGatherL1(const int* ptr, const vint16& idx) {}
FORCEINLINE void prefetchGatherL2(const int* ptr, const vint16& idx) {}
FORCEINLINE void prefetchGatherL1(const vbool16& mask, const int* ptr, const vint16& idx) {}
FORCEINLINE void prefetchGatherL2(const vbool16& mask, const int* ptr, const vint16& idx) {}

// Store functions
// ---------------

FORCEINLINE void ucompress(const vbool16& mask, int* ptr, const vint16& a)
{
    int offset = 0;
    for (int i = 0; i < 16; ++i)
        if (mask.m & (1 << i)) ptr[offset++] = a[i];
}

FORCEINLINE void compress(const vbool16& mask, int* ptr, const vint16& a)
{
    ucompress(mask, ptr, a);
}

FORCEINLINE void store(int* ptr, const vint16& a)
{
    for (int i = 0; i < 4; ++i)
        _mm_store_si128((__m128i*)(ptr + i*4), a.m[i]);
}

FORCEINLINE void ustore(int* ptr, const vint16& a)
{
    for (int i = 0; i < 4; ++i)
        _mm_storeu_si128((__m128i*)(ptr + i*4), a.m[i]);
}

FORCEINLINE void store(const vbool16& mask, int* ptr, const vint16& a)
{
    for (int i = 0; i < 16; ++i)
        if (mask.m & (1 << i)) ptr[i] = a[i];
}

FORCEINLINE void scatter(int* ptr, const vint16& idx, const vint16& a)
{
    for (int i = 0; i < 16; ++i)
        ptr[idx[i]] = a[i];
}

FORCEINLINE void scatter(const vbool16& mask, int* ptr, const vint16& idx, const vint16& a)
{
    for (int i = 0; i < 16; ++i)
        if (mask.m & (1 << i)) ptr[idx[i]] = a[i];
}

// Select function
// ---------------

FORCEINLINE vint16 select(const vbool16& mask, const vint16& a, const vint16& b)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
    {
        __m128i blendMask = *(const __m128i*)(&simdMaskTable4[(mask.m >> (i*4)) & 0xf]);
        r.m[i] = _mm_blendv_epi8(b.m[i], a.m[i], blendMask);
    }
    return r;
}

// Arithmetic operators
// --------------------

FORCEINLINE vint16 operator +(const vint16& a, const vint16& b)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_add_epi32(a.m[i], b.m[i]);
    return r;
}

FORCEINLINE vint16 operator -(const vint16& a, const vint16& b)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_sub_epi32(a.m[i], b.m[i]);
    return r;
}

FORCEINLINE vint16 operator *(const vint16& a, const vint16& b)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_mullo_epi32(a.m[i], b.m[i]);
    return r;
}

// Bitwise operators
// -----------------

FORCEINLINE vint16 operator &(const vint16& a, const vint16& b)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_and_si128(a.m[i], b.m[i]);
    return r;
}

FORCEINLINE vint16 operator |(const vint16& a, const vint16& b)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_or_si128(a.m[i], b.m[i]);
    return r;
}

FORCEINLINE vint16 operator ^(const vint16& a, const vint16& b)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_xor_si128(a.m[i], b.m[i]);
    return r;
}

FORCEINLINE vint16 operator <<(const vint16& a, int b)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_slli_epi32(a.m[i], b);
    return r;
}

FORCEINLINE vint16 operator >>(const vint16& a, int b)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_srai_epi32(a.m[i], b);
    return r;
}

FORCEINLINE vint16 shl(const vint16& a, int b)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_slli_epi32(a.m[i], b);
    return r;
}

FORCEINLINE vint16 shr(const vint16& a, int b)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_srli_epi32(a.m[i], b);
    return r;
}

// Assignment operators
// --------------------

FORCEINLINE vint16& operator +=(vint16& a, const vint16& b) { return a = a + b; }
FORCEINLINE vint16& operator -=(vint16& a, const vint16& b) { return a = a - b; }
FORCEINLINE vint16& operator *=(vint16& a, const vint16& b) { return a = a * b; }
FORCEINLINE vint16& operator &=(vint16& a, const vint16& b) { return a = a & b; }
FORCEINLINE vint16& operator |=(vint16& a, const vint16& b) { return a = a | b; }
FORCEINLINE vint16& operator ^=(vint16& a, const vint16& b) { return a = a ^ b; }

FORCEINLINE vint16& operator <<=(vint16& a, int b) { return a = a << b; }
FORCEINLINE vint16& operator >>=(vint16& a, int b) { return a = a >> b; }

// Compare operators
// -----------------

FORCEINLINE vbool16 operator ==(const vint16& a, const vint16& b)
{
    int mask = 0;
    for (int i = 0; i < 4; ++i)
        mask |= _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(a.m[i], b.m[i]))) << (i*4);
    return mask;
}

FORCEINLINE vbool16 operator <(const vint16& a, const vint16& b)
{
    int mask = 0;
    for (int i = 0; i < 4; ++i)
        mask |= _mm_movemask_ps(_mm_castsi128_ps(_mm_cmplt_epi32(a.m[i], b.m[i]))) << (i*4);
    return mask;
}

FORCEINLINE vbool16 operator >(const vint16& a, const vint16& b)
{
    int mask = 0;
    for (int i = 0; i < 4; ++i)
        mask |= _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(a.m[i], b.m[i]))) << (i*4);
    return mask;
}

FORCEINLINE vbool16 operator !=(const vint16& a, const vint16& b)
{
    return !(a == b);
}

FORCEINLINE vbool16 operator <=(const vint16& a, const vint16& b)
{
    return !(a > b);
}

FORCEINLINE vbool16 operator >=(const vint16& a, const vint16& b)
{
    return !(a < b);
}

// Math functions
// --------------

FORCEINLINE vint16 min(const vint16& a, const vint16& b)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_min_epi32(a.m[i], b.m[i]);
    return r;
}

FORCEINLINE vint16 max(const vint16& a, const vint16& b)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_max_epi32(a.m[i], b.m[i]);
    return r;
}

} // namespace prt
