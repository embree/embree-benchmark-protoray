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
#include "vbool16_avx512.h"

namespace prt {

// vint16
template <>
struct var<int,16>
{
    union
    {
        __m512i m;
        int v[16];
    };

    FORCEINLINE var() {}

    FORCEINLINE var(const vint16& a) : m(a.m) {}
    FORCEINLINE var(__m512i a) : m(a) {}
    FORCEINLINE var(int x) : m(_mm512_set1_epi32(x)) {}
    FORCEINLINE var(int x0, int x1, int x2, int x3) : m(_mm512_set4_epi32(x3, x2, x1, x0)) {}

    FORCEINLINE var(int x0, int x1, int x2,  int x3,  int x4,  int x5,  int x6,  int x7,
                   int x8, int x9, int x10, int x11, int x12, int x13, int x14, int x15)
        : m(_mm512_set_epi32(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1, x0)) {}

    FORCEINLINE var(Zero) : m(_mm512_setzero_epi32()) {}
    FORCEINLINE var(One)  : m(_mm512_set1_epi32(1)) {}
    FORCEINLINE var(Step) : m(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)) {}

    FORCEINLINE vint16& operator =(const vint16& a)
    {
        m = a.m;
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
};

// Load functions
// --------------

template <>
FORCEINLINE vint16 load(const int* ptr)
{
    return _mm512_load_epi32(ptr);
}

template <>
FORCEINLINE vint16 load(const vbool16& mask, const int* ptr)
{
    return _mm512_mask_load_epi32(_mm512_undefined_epi32(), mask.m, ptr);
}

template <>
FORCEINLINE vint16 uload(const int* ptr)
{
    return _mm512_loadu_si512(ptr);
}

FORCEINLINE vint16 uexpand(const vbool16& mask, const int* ptr)
{
    return _mm512_maskz_expandloadu_epi32(mask.m, ptr);
}

FORCEINLINE vint16 gather(const int* ptr, const vint16& idx)
{
    return _mm512_i32gather_epi32(idx.m, ptr, 4);
}

FORCEINLINE vint16 gather(const vbool16& mask, const int* ptr, const vint16& idx)
{
    return _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), mask.m, idx.m, ptr, 4);
}

FORCEINLINE void prefetchGatherL1(const int* ptr, const vint16& idx)
{
#if defined(__AVX512PF__) || defined(__KNC__)
    _mm512_prefetch_i32gather_ps(idx.m, ptr, 4, _MM_HINT_T0);
#endif
}

FORCEINLINE void prefetchGatherL2(const int* ptr, const vint16& idx)
{
#if defined(__AVX512PF__) || defined(__KNC__)
    _mm512_prefetch_i32gather_ps(idx.m, ptr, 4, _MM_HINT_T1);
#endif
}

FORCEINLINE void prefetchGatherL1(const vbool16& mask, const int* ptr, const vint16& idx)
{
#if defined(__AVX512PF__) || defined(__KNC__)
    _mm512_mask_prefetch_i32gather_ps(idx.m, mask.m, ptr, 4, _MM_HINT_T0);
#endif
}

FORCEINLINE void prefetchGatherL2(const vbool16& mask, const int* ptr, const vint16& idx)
{
#if defined(__AVX512PF__) || defined(__KNC__)
    _mm512_mask_prefetch_i32gather_ps(idx.m, mask.m, ptr, 4, _MM_HINT_T1);
#endif
}

// Store functions
// ---------------

FORCEINLINE void store(int* ptr, const vint16& a)
{
    _mm512_store_epi32(ptr, a.m);
}

FORCEINLINE void store(const vbool16& mask, int* ptr, const vint16& a)
{
    _mm512_mask_store_epi32(ptr, mask.m, a.m);
}

FORCEINLINE void ustore(int* ptr, const vint16& a)
{
    _mm512_storeu_si512(ptr, a.m);
}

FORCEINLINE void ucompress(const vbool16& mask, int* ptr, const vint16& a)
{
    _mm512_mask_compressstoreu_epi32(ptr, mask.m, a.m);
}

FORCEINLINE void compress(const vbool16& mask, int* ptr, const vint16& a)
{
    _mm512_mask_compressstoreu_epi32(ptr, mask.m, a.m);
}

FORCEINLINE void scatter(int* ptr, const vint16& idx, const vint16& a)
{
    _mm512_i32scatter_epi32(ptr, idx.m, a.m, 4);
}

FORCEINLINE void scatter(const vbool16& mask, int* ptr, const vint16& idx, const vint16& a)
{
    _mm512_mask_i32scatter_epi32(ptr, mask.m, idx.m, a.m, 4);
}

// Select function
// ---------------

FORCEINLINE vint16 select(const vbool16& mask, const vint16& a, const vint16& b)
{
    return _mm512_mask_mov_epi32(b.m, mask.m, a.m);
}

// Arithmetic operators
// --------------------

FORCEINLINE vint16 operator +(const vint16& a, const vint16& b)
{
    return _mm512_add_epi32(a.m, b.m);
}

FORCEINLINE vint16 operator -(const vint16& a, const vint16& b)
{
    return _mm512_sub_epi32(a.m, b.m);
}

FORCEINLINE vint16 operator *(const vint16& a, const vint16& b)
{
    return _mm512_mullo_epi32(a.m, b.m);
}

// Bitwise operators
// -----------------

FORCEINLINE vint16 operator &(const vint16& a, const vint16& b)
{
    return _mm512_and_epi32(a.m, b.m);
}

FORCEINLINE vint16 operator |(const vint16& a, const vint16& b)
{
    return _mm512_or_epi32(a.m, b.m);
}

FORCEINLINE vint16 operator ^(const vint16& a, const vint16& b)
{
    return _mm512_xor_epi32(a.m, b.m);
}

FORCEINLINE vint16 operator <<(const vint16& a, int b)
{
    return _mm512_slli_epi32(a.m, b);
}

FORCEINLINE vint16 operator >>(const vint16& a, int b)
{
    return _mm512_srai_epi32(a.m, b);
}

FORCEINLINE vint16 shl(const vint16& a, int b)
{
    return _mm512_slli_epi32(a.m, b);
}

FORCEINLINE vint16 shr(const vint16& a, int b)
{
    return _mm512_srli_epi32(a.m, b);
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
    return _mm512_cmp_epi32_mask(a.m, b.m, _MM_CMPINT_EQ);
}

FORCEINLINE vbool16 operator <(const vint16& a, const vint16& b)
{
    return _mm512_cmp_epi32_mask(a.m, b.m, _MM_CMPINT_LT);
}

FORCEINLINE vbool16 operator >(const vint16& a, const vint16& b)
{
    return _mm512_cmp_epi32_mask(a.m, b.m, _MM_CMPINT_GT);
}

FORCEINLINE vbool16 operator !=(const vint16& a, const vint16& b)
{
    return _mm512_cmp_epi32_mask(a.m, b.m, _MM_CMPINT_NE);
}

FORCEINLINE vbool16 operator <=(const vint16& a, const vint16& b)
{
    return _mm512_cmp_epi32_mask(a.m, b.m, _MM_CMPINT_LE);
}

FORCEINLINE vbool16 operator >=(const vint16& a, const vint16& b)
{
    return _mm512_cmp_epi32_mask(a.m, b.m, _MM_CMPINT_GE);
}

// Math functions
// --------------

FORCEINLINE vint16 min(const vint16& a, const vint16& b)
{
    return _mm512_min_epi32(a.m, b.m);
}

FORCEINLINE vint16 max(const vint16& a, const vint16& b)
{
    return _mm512_max_epi32(a.m, b.m);
}

} // namespace prt
