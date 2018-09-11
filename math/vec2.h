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

#include <iostream>
#include "math.h"
#include "simd.h"

namespace prt {

template <class T>
struct Vec2
{
    typedef T Scalar;
    static const int size = 2;

    T x, y;

    FORCEINLINE Vec2() {}

    FORCEINLINE Vec2(Zero) : x(zero), y(zero) {}
    FORCEINLINE Vec2(One) : x(one), y(one) {}

    FORCEINLINE Vec2(const Vec2<T>& a) : x(a.x), y(a.y) {}

    template <class T1>
    FORCEINLINE Vec2(const Vec2<T1>& a) : x(T(a.x)), y(T(a.y)) {}

    FORCEINLINE Vec2(const T& x, const T& y) : x(x), y(y) {}

    FORCEINLINE Vec2(const T& a) : x(a), y(a) {}

    FORCEINLINE Vec2<T>& operator =(const Vec2<T>& a)
    {
        x = a.x;
        y = a.y;
        return *this;
    }

    FORCEINLINE T& operator [](size_t i)
	{
		assert(i < 2);
		return (&x)[i];
	}

    FORCEINLINE const T& operator [](size_t i) const
	{
		assert(i < 2);
		return (&x)[i];
	}

    FORCEINLINE const T* getData() const
	{
		return &x;
	}

    FORCEINLINE T* getData()
	{
		return &x;
	}
};

// Typedefs
// --------

typedef Vec2<float> Vec2f;
typedef Vec2<double> Vec2d;
typedef Vec2<int> Vec2i;
typedef Vec2<uint8_t> Vec2uc;
typedef Vec2<bool> Vec2b;

template <class T, int N = simdSize>
using Vec2v = Vec2<var<T,N>>;

typedef Vec2<vfloat>   Vec2vf;
typedef Vec2<vfloat4>  Vec2vf4;
typedef Vec2<vfloat8>  Vec2vf8;
typedef Vec2<vfloat16> Vec2vf16;

typedef Vec2<vint>   Vec2vi;
typedef Vec2<vint4>  Vec2vi4;
typedef Vec2<vint8>  Vec2vi8;
typedef Vec2<vint16> Vec2vi16;

// Traits
// ------

template <class T>
struct ToFloat<Vec2<T>> { typedef Vec2<ToFloatT<T>> Type; };

template <class T>
struct ToInt<Vec2<T>> { typedef Vec2<ToIntT<T>> Type; };

template <class T>
struct ToDouble<Vec2<T>> { typedef Vec2<ToDoubleT<T>> Type; };

template <class T>
struct ToBool<Vec2<T>> { typedef Vec2<ToBoolT<T>> Type; };

// Conversion functions
// --------------------

template <class T>
FORCEINLINE ToFloatT<Vec2<T>> toFloat(const Vec2<T>& a)
{
    return ToFloatT<Vec2<T>>(toFloat(a.x), toFloat(a.y));
}

template <class T>
FORCEINLINE ToIntT<Vec2<T>> toInt(const Vec2<T>& a)
{
    return ToIntT<Vec2<T>>(toInt(a.x), toInt(a.y));
}

template <class T>
FORCEINLINE ToDoubleT<Vec2<T>> toDouble(const Vec2<T>& a)
{
    return ToDoubleT<Vec2<T>>(toDouble(a.x), toDouble(a.y));
}

template <class T>
FORCEINLINE ToFloatT<Vec2<T>> asFloat(const Vec2<T>& a)
{
    return ToFloatT<Vec2<T>>(asFloat(a.x), asFloat(a.y));
}

template <class T>
FORCEINLINE ToIntT<Vec2<T>> asInt(const Vec2<T>& a)
{
    return ToIntT<Vec2<T>>(asInt(a.x), asInt(a.y));
}

template <class T, int N>
FORCEINLINE Vec2<T> toScalar(const Vec2v<T,N>& a)
{
    return Vec2<T>(toScalar(a.x), toScalar(a.y));
}

// Selection functions
// -------------------

template <class T, class P>
FORCEINLINE Vec2<T> select(const P& p, const Vec2<T>& a, const Vec2<T>& b)
{
    return Vec2<T>(select(p, a.x, b.x), select(p, a.y, b.y));
}

template <class T, class P>
FORCEINLINE void set(const P& p, Vec2<T>& a, const Vec2<T>& b)
{
    set(p, a.x, b.x);
    set(p, a.y, b.y);
}

template <class T, class P>
FORCEINLINE void set(const P& p, Vec2<T>* a, const Vec2<T>& b)
{
    set(p, &a->x, b.x);
    set(p, &a->y, b.y);
}

// Arithmetic operators
// --------------------

template <class T>
FORCEINLINE Vec2<T> operator -(const Vec2<T>& a)
{
    return Vec2<T>(-a.x, -a.y);
}

template <class T>
FORCEINLINE Vec2<T> operator +(const Vec2<T>& a, const Vec2<T>& b)
{
	return Vec2<T>(a.x + b.x, a.y + b.y);
}

template <class T>
FORCEINLINE Vec2<T> operator +(const Vec2<T>& a, const T& b)
{
    return Vec2<T>(a.x + b, a.y + b);
}

template <class T>
FORCEINLINE Vec2<T> operator +(const T& a, const Vec2<T>& b)
{
    return Vec2<T>(a + b.x, a + b.y);
}

template <class T>
FORCEINLINE Vec2<T> operator -(const Vec2<T>& a, const Vec2<T>& b)
{
	return Vec2<T>(a.x - b.x, a.y - b.y);
}

template <class T>
FORCEINLINE Vec2<T> operator -(const Vec2<T>& a, const T& b)
{
    return Vec2<T>(a.x - b, a.y - b);
}

template <class T>
FORCEINLINE Vec2<T> operator -(const T& a, const Vec2<T>& b)
{
    return Vec2<T>(a - b.x, a - b.y);
}

template <class T>
FORCEINLINE Vec2<T> operator *(const Vec2<T>& a, const Vec2<T>& b)
{
	return Vec2<T>(a.x * b.x, a.y * b.y);
}

template <class T>
FORCEINLINE Vec2<T> operator *(const Vec2<T>& a, const T& b)
{
    return Vec2<T>(a.x * b, a.y * b);
}

template <class T>
FORCEINLINE Vec2<T> operator *(const T& a, const Vec2<T>& b)
{
    return Vec2<T>(a * b.x, a * b.y);
}

template <class T>
FORCEINLINE Vec2<T> operator /(const Vec2<T>& a, const Vec2<T>& b)
{
	return Vec2<T>(a.x / b.x, a.y / b.y);
}

template <class T>
FORCEINLINE Vec2<T> operator /(const Vec2<T>& a, const T& b)
{
    return Vec2<T>(a.x / b, a.y / b);
}

template <class T>
FORCEINLINE Vec2<T> operator /(const T& a, const Vec2<T>& b)
{
    return Vec2<T>(a / b.x, a / b.y);
}

// Assignment operators
// --------------------

template <class T> FORCEINLINE Vec2<T>& operator +=(Vec2<T>& a, const Vec2<T>& b) { return a = a + b; }
template <class T> FORCEINLINE Vec2<T>& operator +=(Vec2<T>& a, const T& b)       { return a = a + b; }
template <class T> FORCEINLINE Vec2<T>& operator -=(Vec2<T>& a, const Vec2<T>& b) { return a = a - b; }
template <class T> FORCEINLINE Vec2<T>& operator -=(Vec2<T>& a, const T& b)       { return a = a - b; }
template <class T> FORCEINLINE Vec2<T>& operator *=(Vec2<T>& a, const Vec2<T>& b) { return a = a * b; }
template <class T> FORCEINLINE Vec2<T>& operator *=(Vec2<T>& a, const T& b)       { return a = a * b; }
template <class T> FORCEINLINE Vec2<T>& operator /=(Vec2<T>& a, const Vec2<T>& b) { return a = a / b; }
template <class T> FORCEINLINE Vec2<T>& operator /=(Vec2<T>& a, const T& b)       { return a = a / b; }

// Compare operators
// -----------------

template <class T>
FORCEINLINE bool operator ==(const Vec2<T>& a, const Vec2<T>& b)
{
	return a.x == b.x && a.y == b.y;
}

template <class T>
FORCEINLINE bool operator !=(const Vec2<T>& a, const Vec2<T>& b)
{
	return a.x != b.x || a.y != b.y;
}

// Math functions
// --------------

template <class T>
FORCEINLINE Vec2<T> abs(const Vec2<T>& a)
{
    return Vec2<T>(abs(a.x), abs(a.y));
}

template <class T>
FORCEINLINE Vec2<T> rcp(const Vec2<T>& a)
{
    return Vec2<T>(rcp(a.x), rcp(a.y));
}

template <class T>
FORCEINLINE Vec2<T> rcpSafe(const Vec2<T>& a)
{
    return Vec2<T>(rcpSafe(a.x), rcpSafe(a.y));
}

template <class T>
FORCEINLINE Vec2<T> sqrt(const Vec2<T>& a)
{
    return Vec2<T>(sqrt(a.x), sqrt(a.y));
}

template <class T>
FORCEINLINE Vec2<T> floor(const Vec2<T>& a)
{
    return Vec2<T>(floor(a.x), floor(a.y));
}

template <class T>
FORCEINLINE Vec2<T> min(const Vec2<T>& a, const Vec2<T>& b)
{
	return Vec2<T>(min(a.x, b.x), min(a.y, b.y));
}

template <class T>
FORCEINLINE Vec2<T> max(const Vec2<T>& a, const Vec2<T>& b)
{
	return Vec2<T>(max(a.x, b.x), max(a.y, b.y));
}

template <class T>
FORCEINLINE Vec2<T> clamp(const Vec2<T>& value, const T& minValue, const T& maxValue)
{
	return Vec2<T>(clamp(value.x, minValue, maxValue),
				   clamp(value.y, minValue, maxValue));
}

template <class T>
FORCEINLINE ToBoolT<Vec2<T>> isfinite(const Vec2<T>& a)
{
    return ToBoolT<Vec2<T>>(isfinite(a.x), isfinite(a.y));
}

// Vector math functions
// ---------------------

template <class T>
FORCEINLINE T dot(const Vec2<T>& a, const Vec2<T>& b)
{
    return a.x * b.x + a.y * b.y;
}

template <class T>
FORCEINLINE T lengthSqr(const Vec2<T>& a)
{
    return dot(a, a);
}

template <class T>
FORCEINLINE T length(const Vec2<T>& a)
{
    return sqrt(dot(a, a));
}

template <class T>
FORCEINLINE T lengthRcp(const Vec2<T>& a)
{
    return rsqrt(dot(a, a));
}

template <class T>
FORCEINLINE T lengthRcpSafe(const Vec2<T>& a)
{
    return rsqrtSafe(dot(a, a));
}

template <class T>
FORCEINLINE Vec2<T> normalize(const Vec2<T>& a)
{
    return a * rsqrt(dot(a, a));
}

// Reduction functions
// -------------------

template <class T>
FORCEINLINE T reduceMin(const Vec2<T>& a)
{
    return min(a.x, a.y);
}

template <class T>
FORCEINLINE T reduceMax(const Vec2<T>& a)
{
    return max(a.x, a.y);
}

template <class T>
FORCEINLINE T reduceAdd(const Vec2<T>& a)
{
    return a.x + a.y;
}

template <class T>
FORCEINLINE int selectMin(const Vec2<T>& a)
{
    return a.x < a.y ? 0 : 1;
}

template <class T>
FORCEINLINE int selectMax(const Vec2<T>& a)
{
    return a.x > a.y ? 0 : 1;
}

// Test functions
// --------------

template <class T>
FORCEINLINE bool all(const Vec2<T>& a)
{
    return all(a.x) && all(a.y);
}

template <class T>
FORCEINLINE bool any(const Vec2<T>& a)
{
    return any(a.x) || any(a.y);
}

template <class T>
FORCEINLINE bool none(const Vec2<T>& a)
{
    return none(a.x) && none(a.y);
}

// Stream operators
// ----------------

template <class T>
FORCEINLINE std::ostream& operator <<(std::ostream& osm, const Vec2<T>& a)
{
    osm << a.x << "," << a.y;
    return osm;
}

template <class T>
FORCEINLINE std::istream& operator >>(std::istream& ism, Vec2<T>& a)
{
    ism >> a.x;
    if (ism.peek() == ',')
    {
        ism.ignore();
    }
    else
    {
        a.y = a.x;
        return ism;
    }

    ism >> a.y;
    return ism;
}

// SIMD functions
// --------------

template <class T, int N>
FORCEINLINE Vec2v<T,N> gather2(const var<bool,N>& mask, const T* ptr, const var<int,N>& idx)
{
    return Vec2v<T,N>(gather(mask, ptr,   idx),
                      gather(mask, ptr+1, idx));
}

template <class T, int N>
FORCEINLINE Vec2v<T,N> gather2(const var<bool,N>& mask, const T* ptr0, const T* ptr1, const var<int,N>& idx)
{
    return Vec2v<T,N>(gather(mask, ptr0, idx),
                      gather(mask, ptr1, idx));
}

} // namespace prt
