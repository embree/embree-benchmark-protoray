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

namespace prt {

// vbool8
template <>
struct var<bool,8>
{
    union
    {
        __m256 m;
        int v[8];
    };

    FORCEINLINE var() {}

    FORCEINLINE var(const vbool8& a) : m(a.m) {}
    FORCEINLINE var(__m256 a) : m(a) {}
    FORCEINLINE var(__m256i a) : m(_mm256_castsi256_ps(a)) {}

    FORCEINLINE var(__m128i a0, __m128i a1) : m(_mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(a0), a1, 1))) {}

    FORCEINLINE var(int a)
    {
        assert(a >= 0 && a <= 0xff);

        __m128 lo = *(const __m128*)&simdMaskTable4[a & 0xf];
        __m128 hi = *(const __m128*)&simdMaskTable4[a >> 4];

        m = _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
    }

    FORCEINLINE var(Zero) : m(_mm256_setzero_ps()) {}
    FORCEINLINE var(One)  : m(_mm256_castsi256_ps(_mm256_set1_epi32(0xffffffff))) {}

    FORCEINLINE vbool8& operator =(const vbool8& a)
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

// Conversions
// -----------

FORCEINLINE int toIntMask(const vbool8& a)
{
    return _mm256_movemask_ps(a.m);
}

// Bitwise operators
// -----------------

FORCEINLINE vbool8 operator !(const vbool8& a)
{
    return _mm256_xor_ps(a.m, _mm256_castsi256_ps(_mm256_set1_epi32(0xffffffff)));
}

FORCEINLINE vbool8 operator &(const vbool8& a, const vbool8& b)
{
    return _mm256_and_ps(a.m, b.m);
}

FORCEINLINE vbool8 operator |(const vbool8& a, const vbool8& b)
{
    return _mm256_or_ps(a.m, b.m);
}

FORCEINLINE vbool8 operator ^(const vbool8& a, const vbool8& b)
{
    return _mm256_xor_ps(a.m, b.m);
}

FORCEINLINE vbool8 andn(const vbool8& a, const vbool8& b)
{
    return _mm256_andnot_ps(b.m, a.m);
}

// Assignment operators
// --------------------

FORCEINLINE vbool8& operator &=(vbool8& a, const vbool8& b) { return a = a & b; }
FORCEINLINE vbool8& operator |=(vbool8& a, const vbool8& b) { return a = a | b; }
FORCEINLINE vbool8& operator ^=(vbool8& a, const vbool8& b) { return a = a ^ b; }

// Test functions
// --------------

FORCEINLINE bool all(const vbool8& a)
{
    return _mm256_movemask_ps(a.m) == 0xff;
}

FORCEINLINE bool any(const vbool8& a)
{
    return _mm256_movemask_ps(a.m) != 0;
}

FORCEINLINE bool none(const vbool8& a)
{
    return _mm256_movemask_ps(a.m) == 0;
}

// Get/set functions
// -----------------

FORCEINLINE bool get(const vbool8& a, size_t i)
{
    return a[i];
}

FORCEINLINE void set(vbool8& a, size_t i)
{
    a[i] = -1;
}

FORCEINLINE void clear(vbool8& a, size_t i)
{
    a[i] = 0;
}

} // namespace prt
