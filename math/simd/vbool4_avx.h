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

// vbool4
template <>
struct var<bool,4>
{
    union
    {
        __m128 m;
        int v[4];
    };

    FORCEINLINE var() {}

    FORCEINLINE var(const vbool4& a) : m(a.m) {}
    FORCEINLINE var(__m128 a) : m(a) {}
    FORCEINLINE var(__m128i a) : m(_mm_castsi128_ps(a)) {}

    FORCEINLINE var(int a)
    {
        assert(a >= 0 && a <= 0xf);
        m = *(const __m128*)&simdMaskTable4[a];
    }

    FORCEINLINE var(Zero) : m(_mm_setzero_ps()) {}
    FORCEINLINE var(One)  : m(_mm_castsi128_ps(_mm_set1_epi32(0xffffffff))) {}

    FORCEINLINE vbool4& operator =(const vbool4& a)
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

// Conversions
// -----------

FORCEINLINE int toIntMask(const vbool4& a)
{
    return _mm_movemask_ps(a.m);
}

// Bitwise operators
// -----------------

FORCEINLINE vbool4 operator !(const vbool4& a)
{
    return _mm_xor_ps(a.m, _mm_castsi128_ps(_mm_set1_epi32(0xffffffff)));
}

FORCEINLINE vbool4 operator &(const vbool4& a, const vbool4& b)
{
    return _mm_and_ps(a.m, b.m);
}

FORCEINLINE vbool4 operator |(const vbool4& a, const vbool4& b)
{
    return _mm_or_ps(a.m, b.m);
}

FORCEINLINE vbool4 operator ^(const vbool4& a, const vbool4& b)
{
    return _mm_xor_ps(a.m, b.m);
}

FORCEINLINE vbool4 andn(const vbool4& a, const vbool4& b)
{
    return _mm_andnot_ps(b.m, a.m);
}

// Assignment operators
// --------------------

FORCEINLINE vbool4& operator &=(vbool4& a, const vbool4& b) { return a = a & b; }
FORCEINLINE vbool4& operator |=(vbool4& a, const vbool4& b) { return a = a | b; }
FORCEINLINE vbool4& operator ^=(vbool4& a, const vbool4& b) { return a = a ^ b; }

// Test functions
// --------------

FORCEINLINE bool all(const vbool4& a)
{
    return _mm_movemask_ps(a.m) == 0xf;
}

FORCEINLINE bool any(const vbool4& a)
{
    return _mm_movemask_ps(a.m) != 0;
}

FORCEINLINE bool none(const vbool4& a)
{
    return _mm_movemask_ps(a.m) == 0;
}

// Get/set functions
// -----------------

FORCEINLINE bool get(const vbool4& a, size_t i)
{
    return a[i];
}

FORCEINLINE void set(vbool4& a, size_t i)
{
    a[i] = -1;
}

FORCEINLINE void clear(vbool4& a, size_t i)
{
    a[i] = 0;
}

} // namespace prt
