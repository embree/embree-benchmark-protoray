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

#include "sys/constants.h"
#include "math_common.h"
#include "math_avx.h"

namespace prt {

using std::abs;
using std::sqrt;
using std::pow;
using std::exp;
using std::sin;
using std::cos;
using std::acos;
using std::tan;
using std::atan2;
using std::isfinite;

template <class T>
FORCEINLINE T sqr(const T& x)
{
    return x * x;
}

FORCEINLINE float degToRad(float x)
{
    return x * (float(pi)/180.0f);
}

FORCEINLINE float radToDeg(float x)
{
    return x * (180.0f/float(pi));
}

template <class T>
FORCEINLINE void sincos(const T& x, T& sinResult, T& cosResult)
{
    // FIXME
    sinResult = sin(x);
    cosResult = cos(x);
}

template <class T>
FORCEINLINE T sin2cos(const T& x)
{
    return sqrt(max(T(one)-x*x, T(zero)));
}

template <class T>
FORCEINLINE T cos2sin(const T& x)
{
    return sin2cos(x);
}

template <class T>
FORCEINLINE T clamp(const T& value, const T& minValue, const T& maxValue)
{
    return min(max(value, minValue), maxValue);
}

template <class T, class S>
FORCEINLINE T lerp(const T& a, const T& b, const S& s)
{
    return a + s * (b - a);
}

template <class T>
FORCEINLINE T boxStep(const T& min, const T& val)
{
    return select(val < min, T(zero), T(one));
}

template <class T>
FORCEINLINE T smoothStep(const T& edge0, const T& edge1, const T& x)
{
    T t = clamp((x - edge0) / (edge1 - edge0), T(0.0f), T(1.0f));
    return t * t * (T(3.0f) - T(2.0f) * t);
}

// Does not return infinity for 0!
FORCEINLINE float rcpSafe(float x)
{
    return x == 0.0f ? posMax : rcp(x);
}

// Does not return NaN
template <class T>
FORCEINLINE T sqrtSafe(const T& x)
{
    return sqrt(max(x, T(zero)));
}

// Does not return NaN
template <class T>
FORCEINLINE T acosSafe(const T& x)
{
    return acos(clamp(x, -T(one), T(one)));
}

FORCEINLINE int shl(int a, int b)
{
    return uint(a) << uint(b);
}

FORCEINLINE int shr(int a, int b)
{
    return uint(a) >> uint(b);
}

FORCEINLINE int bitTestAndComplement(int value, int index)
{
#if defined(_WIN32)
    long r = value;
    _bittestandcomplement(&r, index);
    return r;
#else
    int r = 0;
    asm("btc %1,%0" : "=r"(r) : "r"(index), "0"(value) : "flags");
    return r;
#endif
}

FORCEINLINE int bitScanAndComplement(int& value)
{
    int i = bitScan(value);
    value &= value-1; // BLSR
    return i;
}

FORCEINLINE bool isPowerOfTwo(int x)
{
    return (x & (x-1)) == 0;
}

// Conversion functions
// --------------------

template <class T>
FORCEINLINE float toFloat(const T& x) { return T(x); }

FORCEINLINE float toFloatUnorm(int x) { return float(uint32_t(x)) * 0x1.0p-32f; }

template <class T>
FORCEINLINE int toInt(const T& x) { return T(x); }

template <class T>
FORCEINLINE double toDouble(const T& x) { return T(x); }

FORCEINLINE float asFloat(float x) { return x; }
FORCEINLINE float asFloat(int x) { return bitwise_cast<float>(x); }
FORCEINLINE int asInt(int x) { return x; }
FORCEINLINE int asInt(float x) { return bitwise_cast<int>(x); }

FORCEINLINE int toIntSafe(float x)
{
    int xi = toInt(max(x, -2147483648.0f));
    xi = select(x >= 2147483648.0f, 2147483647, xi);
    return xi;
}

// Reduction functions
// -------------------

template <class T>
FORCEINLINE T reduceMin(const T& x) { return x; }

template <class T>
FORCEINLINE T reduceMax(const T& x) { return x; }

template <class T>
FORCEINLINE T reduceAdd(const T& x) { return x; }

} // namespace prt
