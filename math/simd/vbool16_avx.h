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

// vbool16
template <>
struct var<bool,16>
{
    uint16_t m;

    FORCEINLINE var() {}
    FORCEINLINE var(int a) : m((uint16_t)a) {}

    FORCEINLINE var(Zero) : m(0) {}
    FORCEINLINE var(One)  : m(0xffff) {}

    FORCEINLINE operator int() const
    {
        return m;
    }
};

// Conversions
// -----------

FORCEINLINE int toIntMask(const vbool16& a)
{
    return a.m;
}

// Bitwise operators
// -----------------

FORCEINLINE vbool16 operator !(const vbool16& a)
{
    return ~a.m;
}

FORCEINLINE vbool16 operator &(const vbool16& a, const vbool16& b)
{
    return a.m & b.m;
}

FORCEINLINE vbool16 operator |(const vbool16& a, const vbool16& b)
{
    return a.m | b.m;
}

FORCEINLINE vbool16 operator ^(const vbool16& a, const vbool16& b)
{
    return a.m ^ b.m;
}

FORCEINLINE vbool16 andn(const vbool16& a, const vbool16& b)
{
    return a.m & ~b.m;
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
    return a.m == 0xffff;
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
