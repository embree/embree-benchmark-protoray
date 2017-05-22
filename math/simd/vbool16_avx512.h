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

namespace prt {

// vbool16
template <>
struct var<bool,16>
{
    __mmask16 m;

    FORCEINLINE var() {}

    FORCEINLINE var(const vbool16& a) : m(a.m) {}
    FORCEINLINE var(__mmask16 a) : m(a) {}
    FORCEINLINE var(int a) : m(_mm512_int2mask(a)) {}

    FORCEINLINE var(Zero) : m(_mm512_int2mask(0)) {}
    FORCEINLINE var(One)  : m(_mm512_int2mask(0xffff)) {}

    FORCEINLINE vbool16& operator =(const vbool16& a)
    {
        m = a.m;
        return *this;
    }

    FORCEINLINE operator int() const
    {
        return _mm512_mask2int(m);
    }
};

// Conversions
// -----------

FORCEINLINE int toIntMask(const vbool16& a)
{
    return _mm512_mask2int(a.m);
}

// Bitwise operators
// -----------------

FORCEINLINE vbool16 operator !(const vbool16& a)
{
    return _mm512_knot(a.m);
}

FORCEINLINE vbool16 operator &(const vbool16& a, const vbool16& b)
{
    return _mm512_kand(a.m, b.m);
}

FORCEINLINE vbool16 operator |(const vbool16& a, const vbool16& b)
{
    return _mm512_kor(a.m, b.m);
}

FORCEINLINE vbool16 operator ^(const vbool16& a, const vbool16& b)
{
    return _mm512_kxor(a.m, b.m);
}

FORCEINLINE vbool16 andn(const vbool16& a, const vbool16& b)
{
    return _mm512_kandn(b.m, a.m);
}

// Assignment operators
// --------------------

FORCEINLINE vbool16& operator &=(vbool16& a, const vbool16& b) { return a = a & b; }
FORCEINLINE vbool16& operator |=(vbool16& a, const vbool16& b) { return a = a | b; }
FORCEINLINE vbool16& operator ^=(vbool16& a, const vbool16& b) { return a = a ^ b; }

// Test functions
// --------------

FORCEINLINE bool all(const vbool16& a)
{
    return _mm512_kortestc(a.m, a.m);
}

FORCEINLINE bool any(const vbool16& a)
{
    return a.m != 0;
}

FORCEINLINE bool none(const vbool16& a)
{
    return a.m == 0;
}

// Get/set functions
// -----------------

FORCEINLINE bool get(const vbool16& a, size_t i)
{
    assert(i < 16);
    return (toIntMask(a) >> int(i)) & 1;
}

FORCEINLINE void set(vbool16& a, size_t i)
{
    assert(i < 16);
    a |= 1 << int(i);
}

FORCEINLINE void clear(vbool16& a, size_t i)
{
    assert(i < 16);
    a = andn(a, 1 << int(i));
}

} // namespace prt
