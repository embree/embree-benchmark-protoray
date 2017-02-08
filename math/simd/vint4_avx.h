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

#include <immintrin.h>
#include "../simd_common.h"
#include "vbool4_avx.h"

namespace prt {

// vint4
template <>
struct var<int,4>
{
    union
    {
        __m128i m;
        int v[4];
    };

    FORCEINLINE var() {}

    FORCEINLINE var(const vint4& a) : m(a.m) {}
    FORCEINLINE var(__m128i a) : m(a) {}
    FORCEINLINE var(int a) : m(_mm_set1_epi32(a)) {}
    FORCEINLINE var(int a0, int a1, int a2, int a3) : m(_mm_set_epi32(a3, a2, a1, a0)) {}

    FORCEINLINE var(Zero) : m(_mm_setzero_si128()) {}
    FORCEINLINE var(One)  : m(_mm_set1_epi32(1)) {}
    FORCEINLINE var(Step) : m(_mm_set_epi32(3, 2, 1, 0)) {}

    FORCEINLINE vint4& operator =(const vint4& a)
    {
        m = a.m;
        return *this;
    }

    FORCEINLINE const int& operator [](size_t i) const
	{
		assert(i < 4);
        return v[i];
	}

    FORCEINLINE int& operator [](size_t i)
	{
		assert(i < 4);
        return v[i];
	}
};

// Load functions
// --------------

template <>
FORCEINLINE vint4 load(const int* ptr)
{
    return _mm_load_si128((const __m128i*)ptr);
}

template <>
FORCEINLINE vint4 load(const vbool4& mask, const int* ptr)
{
    vint4 r;
    for (int i = 0; i < 4; ++i)
        if (mask[i]) r[i] = ptr[i];
    return r;
}

template <>
FORCEINLINE vint4 uload(const int* ptr)
{
    return _mm_loadu_si128((const __m128i*)ptr);
}

FORCEINLINE vint4 gather(const int* ptr, const vint4& idx)
{
    vint4 r;
    for (int i = 0; i < 4; ++i)
        r[i] = ptr[idx[i]];
    return r;
}

FORCEINLINE vint4 gather(const vbool4& mask, const int* ptr, const vint4& idx)
{
    vint4 r;
    for (int i = 0; i < 4; ++i)
        if (mask[i]) r[i] = ptr[idx[i]];
    return r;
}

FORCEINLINE void prefetchGatherL1(const int* ptr, const vint4& idx) {}
FORCEINLINE void prefetchGatherL2(const int* ptr, const vint4& idx) {}
FORCEINLINE void prefetchGatherL1(const vbool4& mask, const int* ptr, const vint4& idx) {}
FORCEINLINE void prefetchGatherL2(const vbool4& mask, const int* ptr, const vint4& idx) {}

// Store functions
// ---------------

FORCEINLINE void store(int* ptr, const vint4& a)
{
    _mm_store_si128((__m128i*)ptr, a.m);
}

FORCEINLINE void store(const vbool4& mask, int* ptr, const vint4& a)
{
    for (int i = 0; i < 4; ++i)
        if (mask[i]) ptr[i] = a[i];
}

FORCEINLINE void ucompress(const vbool4& mask, int* ptr, const vint4& a)
{
    int offset = 0;
    for (int i = 0; i < 4; ++i)
        if (mask[i]) ptr[offset++] = a[i];
}

FORCEINLINE void scatter(int* ptr, const vint4& idx, const vint4& a)
{
    for (int i = 0; i < 4; ++i)
        ptr[idx[i]] = a[i];
}

FORCEINLINE void scatter(const vbool4& mask, int* ptr, const vint4& idx, const vint4& a)
{
    for (int i = 0; i < 4; ++i)
        if (mask[i]) ptr[idx[i]] = a[i];
}

// Select function
// ---------------

FORCEINLINE vint4 select(const vbool4& mask, const vint4& a, const vint4& b)
{
    return _mm_blendv_epi8(b.m, a.m, _mm_castps_si128(mask.m));
}

// Shuffle functions
// -----------------

template <int i>
FORCEINLINE int extract(const vint4& a)
{
    return _mm_extract_epi32(a.m, i);
}

template <int i>
FORCEINLINE vint4 broadcast(const vint4& a)
{
    return _mm_shuffle_epi32(a.m, _MM_SHUFFLE(i, i, i, i));
}

template <int i0, int i1, int i2, int i3>
FORCEINLINE vint4 permute4(const vint4& a)
{
    return _mm_shuffle_epi32(a.m, _MM_SHUFFLE(i3, i2, i1, i0));
}

// Arithmetic operators
// --------------------

FORCEINLINE vint4 operator +(const vint4& a, const vint4& b)
{
    return _mm_add_epi32(a.m, b.m);
}

FORCEINLINE vint4 operator -(const vint4& a, const vint4& b)
{
    return _mm_sub_epi32(a.m, b.m);
}

FORCEINLINE vint4 operator *(const vint4& a, const vint4& b)
{
    return _mm_mullo_epi32(a.m, b.m);
}

// Bitwise operators
// -----------------

FORCEINLINE vint4 operator &(const vint4& a, const vint4& b)
{
    return _mm_and_si128(a.m, b.m);
}

FORCEINLINE vint4 operator |(const vint4& a, const vint4& b)
{
    return _mm_or_si128(a.m, b.m);
}

FORCEINLINE vint4 operator ^(const vint4& a, const vint4& b)
{
    return _mm_xor_si128(a.m, b.m);
}

FORCEINLINE vint4 andn(const vint4& a, const vint4& b)
{
    return _mm_andnot_si128(b.m, a.m);
}

FORCEINLINE vint4 operator <<(const vint4& a, int b)
{
    return _mm_slli_epi32(a.m, b);
}

FORCEINLINE vint4 operator >>(const vint4& a, int b)
{
    return _mm_srai_epi32(a.m, b);
}

FORCEINLINE vint4 shl(const vint4& a, int b)
{
    return _mm_slli_epi32(a.m, b);
}

FORCEINLINE vint4 shr(const vint4& a, int b)
{
    return _mm_srli_epi32(a.m, b);
}

// Assignment operators
// --------------------

FORCEINLINE vint4& operator +=(vint4& a, const vint4& b) { return a = a + b; }
FORCEINLINE vint4& operator -=(vint4& a, const vint4& b) { return a = a - b; }
FORCEINLINE vint4& operator *=(vint4& a, const vint4& b) { return a = a * b; }
FORCEINLINE vint4& operator &=(vint4& a, const vint4& b) { return a = a & b; }
FORCEINLINE vint4& operator |=(vint4& a, const vint4& b) { return a = a | b; }
FORCEINLINE vint4& operator ^=(vint4& a, const vint4& b) { return a = a ^ b; }

FORCEINLINE vint4& operator <<=(vint4& a, int b) { return a = a << b; }
FORCEINLINE vint4& operator >>=(vint4& a, int b) { return a = a >> b; }

// Compare operators
// -----------------

FORCEINLINE vbool4 operator ==(const vint4& a, const vint4& b)
{
    return _mm_cmpeq_epi32(a.m, b.m);
}

FORCEINLINE vbool4 operator <(const vint4& a, const vint4& b)
{
    return _mm_cmplt_epi32(a.m, b.m);
}

FORCEINLINE vbool4 operator >(const vint4& a, const vint4& b)
{
    return _mm_cmpgt_epi32(a.m, b.m);
}

FORCEINLINE vbool4 operator !=(const vint4& a, const vint4& b)
{
    return !(a == b);
}

FORCEINLINE vbool4 operator <=(const vint4& a, const vint4& b)
{
    return !(a > b);
}

FORCEINLINE vbool4 operator >=(const vint4& a, const vint4& b)
{
    return !(a < b);
}

// Compare functions
// -----------------

FORCEINLINE vbool4 cmpEq(const vint4& a, const vint4& b) { return a == b; }
FORCEINLINE vbool4 cmpNe(const vint4& a, const vint4& b) { return a != b; }
FORCEINLINE vbool4 cmpLt(const vint4& a, const vint4& b) { return a <  b; }
FORCEINLINE vbool4 cmpLe(const vint4& a, const vint4& b) { return a <= b; }
FORCEINLINE vbool4 cmpGt(const vint4& a, const vint4& b) { return a >  b; }
FORCEINLINE vbool4 cmpGe(const vint4& a, const vint4& b) { return a >= b; }

FORCEINLINE vbool4 cmpEq(const vbool4& mask, const vint4& a, const vint4& b) { return (a == b) & mask; }
FORCEINLINE vbool4 cmpNe(const vbool4& mask, const vint4& a, const vint4& b) { return (a != b) & mask; }
FORCEINLINE vbool4 cmpLt(const vbool4& mask, const vint4& a, const vint4& b) { return (a <  b) & mask; }
FORCEINLINE vbool4 cmpLe(const vbool4& mask, const vint4& a, const vint4& b) { return (a <= b) & mask; }
FORCEINLINE vbool4 cmpGt(const vbool4& mask, const vint4& a, const vint4& b) { return (a >  b) & mask; }
FORCEINLINE vbool4 cmpGe(const vbool4& mask, const vint4& a, const vint4& b) { return (a >= b) & mask; }

// Math functions
// --------------

FORCEINLINE vint4 min(const vint4& a, const vint4& b)
{
    return _mm_min_epi32(a.m, b.m);
}

FORCEINLINE vint4 max(const vint4& a, const vint4& b)
{
    return _mm_max_epi32(a.m, b.m);
}

} // namespace prt
