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
#include "vbool8_avx.h"

namespace prt {

// vint8
template <>
struct var<int,8>
{
    union
    {
        __m256i m;
        int v[8];
    };

    FORCEINLINE var() {}

    FORCEINLINE var(const vint8& a) : m(a.m) {}
    FORCEINLINE var(__m256i a) : m(a) {}
    FORCEINLINE var(int a) : m(_mm256_set1_epi32(a)) {}
    FORCEINLINE var(int a0, int a1, int a2, int a3, int a4, int a5, int a6, int a7) : m(_mm256_set_epi32(a7, a6, a5, a4, a3, a2, a1, a0)) {}

    FORCEINLINE var(Zero) : m(_mm256_setzero_si256()) {}
    FORCEINLINE var(One)  : m(_mm256_set1_epi32(1)) {}
    FORCEINLINE var(Step) : m(_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)) {}

    FORCEINLINE vint8& operator =(const vint8& a)
    {
        m = a.m;
        return *this;
    }

    FORCEINLINE const int& operator [](size_t i) const
    {
        assert(i < 8);
        return v[i];
    }

    FORCEINLINE int& operator [](size_t i)
    {
        assert(i < 8);
        return v[i];
    }
};

// Load functions
// --------------

template <>
FORCEINLINE vint8 load(const int* ptr)
{
    return _mm256_load_si256((const __m256i*)ptr);
}

template <>
FORCEINLINE vint8 load(const vbool8& mask, const int* ptr)
{
    return _mm256_maskload_epi32(ptr, _mm256_castps_si256(mask.m));
}

template <>
FORCEINLINE vint8 uload(const int* ptr)
{
    return _mm256_loadu_si256((const __m256i*)ptr);
}

FORCEINLINE vint8 uexpand(const vbool8& mask, const int* ptr)
{
    __m256i key = _mm256_load_si256((const __m256i*)&simdExpandTable8[_mm256_movemask_ps(mask.m)]);
    return _mm256_permutevar8x32_epi32(_mm256_maskload_epi32(ptr, key), key);
}

FORCEINLINE vint8 gather(const int* ptr, const vint8& idx)
{
    return _mm256_i32gather_epi32(ptr, idx.m, 4);
}

FORCEINLINE vint8 gather(const vbool8& mask, const int* ptr, const vint8& idx)
{
    return _mm256_mask_i32gather_epi32(_mm256_undefined_si256(), ptr, idx.m, _mm256_castps_si256(mask.m), 4);
}

FORCEINLINE void prefetchGatherL1(const int* ptr, const vint8& idx) {}
FORCEINLINE void prefetchGatherL2(const int* ptr, const vint8& idx) {}
FORCEINLINE void prefetchGatherL1(const vbool8& mask, const int* ptr, const vint8& idx) {}
FORCEINLINE void prefetchGatherL2(const vbool8& mask, const int* ptr, const vint8& idx) {}

// Store functions
// ---------------

FORCEINLINE void store(int* ptr, const vint8& a)
{
    _mm256_store_si256((__m256i*)ptr, a.m);
}

FORCEINLINE void store(const vbool8& mask, int* ptr, const vint8& a)
{
    _mm256_maskstore_epi32(ptr, _mm256_castps_si256(mask.m), a.m);
}

FORCEINLINE void ustore(int* ptr, const vint8& a)
{
    _mm256_storeu_si256((__m256i*)ptr, a.m);
}

FORCEINLINE void ucompress(const vbool8& mask, int* ptr, const vint8& a)
{
    __m256i key = _mm256_load_si256((const __m256i*)&simdCompressTable8[_mm256_movemask_ps(mask.m)]);
    _mm256_maskstore_epi32(ptr, key, _mm256_permutevar8x32_epi32(a.m, key));
}

FORCEINLINE void scatter(int* ptr, const vint8& idx, const vint8& a)
{
    for (int i = 0; i < 8; ++i)
        ptr[idx[i]] = a[i];
}

FORCEINLINE void scatter(const vbool8& mask, int* ptr, const vint8& idx, const vint8& a)
{
    for (int i = 0; i < 8; ++i)
        if (mask[i]) ptr[idx[i]] = a[i];
}

// Select function
// ---------------

FORCEINLINE vint8 select(const vbool8& mask, const vint8& a, const vint8& b)
{
    return _mm256_blendv_epi8(b.m, a.m, _mm256_castps_si256(mask.m));
}

// Arithmetic operators
// --------------------

FORCEINLINE vint8 operator +(const vint8& a, const vint8& b)
{
    return _mm256_add_epi32(a.m, b.m);
}

FORCEINLINE vint8 operator -(const vint8& a, const vint8& b)
{
    return _mm256_sub_epi32(a.m, b.m);
}

FORCEINLINE vint8 operator *(const vint8& a, const vint8& b)
{
    return _mm256_mullo_epi32(a.m, b.m);
}

// Bitwise operators
// -----------------

FORCEINLINE vint8 operator &(const vint8& a, const vint8& b)
{
     return _mm256_and_si256(a.m, b.m);
}

FORCEINLINE vint8 operator |(const vint8& a, const vint8& b)
{
     return _mm256_or_si256(a.m, b.m);
}

FORCEINLINE vint8 operator ^(const vint8& a, const vint8& b)
{
     return _mm256_xor_si256(a.m, b.m);
}

FORCEINLINE vint8 andn(const vint8& a, const vint8& b)
{
    return _mm256_andnot_si256(b.m, a.m);
}

FORCEINLINE vint8 operator <<(const vint8& a, int b)
{
    return _mm256_slli_epi32(a.m, b);
}

FORCEINLINE vint8 operator >>(const vint8& a, int b)
{
    return _mm256_srai_epi32(a.m, b);
}

FORCEINLINE vint8 shl(const vint8& a, int b)
{
    return _mm256_slli_epi32(a.m, b);
}

FORCEINLINE vint8 shr(const vint8& a, int b)
{
    return _mm256_srli_epi32(a.m, b);
}

// Assignment operators
// --------------------

FORCEINLINE vint8& operator +=(vint8& a, const vint8& b) { return a = a + b; }
FORCEINLINE vint8& operator -=(vint8& a, const vint8& b) { return a = a - b; }
FORCEINLINE vint8& operator *=(vint8& a, const vint8& b) { return a = a * b; }
FORCEINLINE vint8& operator &=(vint8& a, const vint8& b) { return a = a & b; }
FORCEINLINE vint8& operator |=(vint8& a, const vint8& b) { return a = a | b; }
FORCEINLINE vint8& operator ^=(vint8& a, const vint8& b) { return a = a ^ b; }

FORCEINLINE vint8& operator <<=(vint8& a, int b) { return a = a << b; }
FORCEINLINE vint8& operator >>=(vint8& a, int b) { return a = a >> b; }

// Compare operators
// -----------------

FORCEINLINE vbool8 operator ==(const vint8& a, const vint8& b)
{
    return _mm256_cmpeq_epi32(a.m, b.m);
}

FORCEINLINE vbool8 operator <(const vint8& a, const vint8& b)
{
    return _mm256_cmpgt_epi32(b.m, a.m);
}

FORCEINLINE vbool8 operator >(const vint8& a, const vint8& b)
{
    return _mm256_cmpgt_epi32(a.m, b.m);
}

FORCEINLINE vbool8 operator !=(const vint8& a, const vint8& b)
{
    return !(a == b);
}

FORCEINLINE vbool8 operator <=(const vint8& a, const vint8& b)
{
    return !(a > b);
}

FORCEINLINE vbool8 operator >=(const vint8& a, const vint8& b)
{
    return !(a < b);
}

// Compare functions
// -----------------

FORCEINLINE vbool8 cmpEq(const vint8& a, const vint8& b) { return a == b; }
FORCEINLINE vbool8 cmpNe(const vint8& a, const vint8& b) { return a != b; }
FORCEINLINE vbool8 cmpLt(const vint8& a, const vint8& b) { return a <  b; }
FORCEINLINE vbool8 cmpLe(const vint8& a, const vint8& b) { return a <= b; }
FORCEINLINE vbool8 cmpGt(const vint8& a, const vint8& b) { return a >  b; }
FORCEINLINE vbool8 cmpGe(const vint8& a, const vint8& b) { return a >= b; }

FORCEINLINE vbool8 cmpEq(const vbool8& mask, const vint8& a, const vint8& b) { return (a == b) & mask; }
FORCEINLINE vbool8 cmpNe(const vbool8& mask, const vint8& a, const vint8& b) { return (a != b) & mask; }
FORCEINLINE vbool8 cmpLt(const vbool8& mask, const vint8& a, const vint8& b) { return (a <  b) & mask; }
FORCEINLINE vbool8 cmpLe(const vbool8& mask, const vint8& a, const vint8& b) { return (a <= b) & mask; }
FORCEINLINE vbool8 cmpGt(const vbool8& mask, const vint8& a, const vint8& b) { return (a >  b) & mask; }
FORCEINLINE vbool8 cmpGe(const vbool8& mask, const vint8& a, const vint8& b) { return (a >= b) & mask; }

// Math functions
// --------------

FORCEINLINE vint8 min(const vint8& a, const vint8& b)
{
    return _mm256_min_epi32(a.m, b.m);
}

FORCEINLINE vint8 max(const vint8& a, const vint8& b)
{
    return _mm256_max_epi32(a.m, b.m);
}

} // namespace prt
