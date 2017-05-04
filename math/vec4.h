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

#include <iostream>
#include "math.h"
#include "simd.h"
#include "vec3.h"

namespace prt {

template <class T>
struct Vec4
{
    typedef T Scalar;
    static const int size = 4;

    T x, y, z, w;

    FORCEINLINE Vec4() {}

    FORCEINLINE Vec4(Zero) : x(zero), y(zero), z(zero), w(zero) {}
    FORCEINLINE Vec4(One) : x(one), y(one), z(one), w(one) {}

    FORCEINLINE Vec4(const Vec4<T>& a) : x(a.x), y(a.y), z(a.z), w(a.w) {}

    template <class T1>
    FORCEINLINE Vec4(const Vec4<T1>& a) : x(T(a.x)), y(T(a.y)), z(T(a.z)), w(T(a.w)) {}

    FORCEINLINE Vec4(const T& x, const T& y, const T& z, const T& w) : x(x), y(y), z(z), w(w) {}

    FORCEINLINE Vec4(const Vec3<T>& xyz, const T& w) : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {}

    FORCEINLINE Vec4(const T& a) : x(a), y(a), z(a), w(a) {}

    FORCEINLINE Vec4<T>& operator =(const Vec4<T>& a)
    {
        x = a.x;
        y = a.y;
        z = a.z;
        w = a.w;
        return *this;
    }

    FORCEINLINE T& operator [](size_t i)
	{
		assert(i < 4);
		return (&x)[i];
	}

    FORCEINLINE const T& operator [](size_t i) const
	{
		assert(i < 4);
		return (&x)[i];
	}

    FORCEINLINE Vec3<T> xyz() const
    {
        return Vec3<T>(x, y, z);
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

typedef Vec4<float> Vec4f;
typedef Vec4<double> Vec4d;
typedef Vec4<int> Vec4i;
typedef Vec4<uint8_t> Vec4uc;
typedef Vec4<bool> Vec4b;

template <class T, int N = simdSize>
using Vec4v = Vec4<var<T,N>>;

typedef Vec4<vfloat>   Vec4vf;
typedef Vec4<vfloat4>  Vec4vf4;
typedef Vec4<vfloat8>  Vec4vf8;
typedef Vec4<vfloat16> Vec4vf16;

typedef Vec4<vint>   Vec4vi;
typedef Vec4<vint4>  Vec4vi4;
typedef Vec4<vint8>  Vec4vi8;
typedef Vec4<vint16> Vec4vi16;

// Traits
// ------

template <class T>
struct ToFloat<Vec4<T>> { typedef Vec4<ToFloatT<T>> Type; };

template <class T>
struct ToInt<Vec4<T>> { typedef Vec4<ToIntT<T>> Type; };

template <class T>
struct ToDouble<Vec4<T>> { typedef Vec4<ToDoubleT<T>> Type; };

template <class T>
struct ToBool<Vec4<T>> { typedef Vec4<ToBoolT<T>> Type; };

// Conversion functions
// --------------------

template <class T>
FORCEINLINE ToFloatT<Vec4<T>> toFloat(const Vec4<T>& a)
{
    return ToFloatT<Vec4<T>>(toFloat(a.x), toFloat(a.y), toFloat(a.z), toFloat(a.w));
}

template <class T>
FORCEINLINE ToIntT<Vec4<T>> toInt(const Vec4<T>& a)
{
    return ToIntT<Vec4<T>>(toInt(a.x), toInt(a.y), toInt(a.z), toInt(a.w));
}

template <class T>
FORCEINLINE ToDoubleT<Vec4<T>> toDouble(const Vec4<T>& a)
{
    return ToDoubleT<Vec4<T>>(toDouble(a.x), toDouble(a.y), toDouble(a.z), toDouble(a.w));
}

template <class T>
FORCEINLINE ToFloatT<Vec4<T>> asFloat(const Vec4<T>& a)
{
    return ToFloatT<Vec4<T>>(asFloat(a.x), asFloat(a.y), asFloat(a.z), asFloat(a.w));
}

template <class T>
FORCEINLINE ToIntT<Vec4<T>> asInt(const Vec4<T>& a)
{
    return ToIntT<Vec4<T>>(asInt(a.x), asInt(a.y), asInt(a.z), asInt(a.w));
}

// Selection functions
// -------------------

template <class T, class P>
FORCEINLINE Vec4<T> select(const P& p, const Vec4<T>& a, const Vec4<T>& b)
{
    return Vec4<T>(select(p, a.x, b.x), select(p, a.y, b.y), select(p, a.z, b.z), select(p, a.w, b.w));
}

template <class T, class P>
FORCEINLINE void set(const P& p, Vec4<T>& a, const Vec4<T>& b)
{
    set(p, a.x, b.x);
    set(p, a.y, b.y);
    set(p, a.z, b.z);
    set(p, a.w, b.w);
}

template <class T, class P>
FORCEINLINE void set(const P& p, Vec4<T>* a, const Vec4<T>& b)
{
    set(p, &a->x, b.x);
    set(p, &a->y, b.y);
    set(p, &a->z, b.z);
    set(p, &a->w, b.w);
}

// Arithmetic operators
// --------------------

template <class T>
FORCEINLINE Vec4<T> operator -(const Vec4<T>& a)
{
    return Vec4<T>(-a.x, -a.y, -a.z, -a.w);
}

template <class T>
FORCEINLINE Vec4<T> operator +(const Vec4<T>& a, const Vec4<T>& b)
{
	return Vec4<T>(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template <class T>
FORCEINLINE Vec4<T> operator +(const Vec4<T>& a, const T& b)
{
    return Vec4<T>(a.x + b, a.y + b, a.z + b, a.w + b);
}

template <class T>
FORCEINLINE Vec4<T> operator +(const T& a, const Vec4<T>& b)
{
    return Vec4<T>(a + b.x, a + b.y, a + b.z, a + b.w);
}

template <class T>
FORCEINLINE Vec4<T> operator -(const Vec4<T>& a, const Vec4<T>& b)
{
	return Vec4<T>(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

template <class T>
FORCEINLINE Vec4<T> operator -(const Vec4<T>& a, const T& b)
{
    return Vec4<T>(a.x - b, a.y - b, a.z - b, a.w - b);
}

template <class T>
FORCEINLINE Vec4<T> operator -(const T& a, const Vec4<T>& b)
{
    return Vec4<T>(a - b.x, a - b.y, a - b.z, a - b.w);
}

template <class T>
FORCEINLINE Vec4<T> operator *(const Vec4<T>& a, const Vec4<T>& b)
{
	return Vec4<T>(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

template <class T>
FORCEINLINE Vec4<T> operator *(const Vec4<T>& a, const T& b)
{
    return Vec4<T>(a.x * b, a.y * b, a.z * b, a.w * b);
}

template <class T>
FORCEINLINE Vec4<T> operator *(const T& a, const Vec4<T>& b)
{
    return Vec4<T>(a * b.x, a * b.y, a * b.z, a * b.w);
}

template <class T>
FORCEINLINE Vec4<T> operator /(const Vec4<T>& a, const Vec4<T>& b)
{
	return Vec4<T>(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

template <class T>
FORCEINLINE Vec4<T> operator /(const Vec4<T>& a, const T& b)
{
    return Vec4<T>(a.x / b, a.y / b, a.z / b, a.w / b);
}

template <class T>
FORCEINLINE Vec4<T> operator /(const T& a, const Vec4<T>& b)
{
    return Vec4<T>(a / b.x, a / b.y, a / b.z, a / b.w);
}

// Assignment operators
// --------------------

template <class T> FORCEINLINE Vec4<T>& operator +=(Vec4<T>& a, const Vec4<T>& b) { return a = a + b; }
template <class T> FORCEINLINE Vec4<T>& operator +=(Vec4<T>& a, const T& b)       { return a = a + b; }
template <class T> FORCEINLINE Vec4<T>& operator -=(Vec4<T>& a, const Vec4<T>& b) { return a = a - b; }
template <class T> FORCEINLINE Vec4<T>& operator -=(Vec4<T>& a, const T& b)       { return a = a - b; }
template <class T> FORCEINLINE Vec4<T>& operator *=(Vec4<T>& a, const Vec4<T>& b) { return a = a * b; }
template <class T> FORCEINLINE Vec4<T>& operator *=(Vec4<T>& a, const T& b)       { return a = a * b; }
template <class T> FORCEINLINE Vec4<T>& operator /=(Vec4<T>& a, const Vec4<T>& b) { return a = a / b; }
template <class T> FORCEINLINE Vec4<T>& operator /=(Vec4<T>& a, const T& b)       { return a = a / b; }

// Compare operators
// -----------------

template <class T>
FORCEINLINE bool operator ==(const Vec4<T>& a, const Vec4<T>& b)
{
	return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

template <class T>
FORCEINLINE bool operator !=(const Vec4<T>& a, const Vec4<T>& b)
{
	return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}

// Math functions
// --------------

template <class T>
FORCEINLINE Vec4<T> abs(const Vec4<T>& a)
{
    return Vec4<T>(abs(a.x), abs(a.y), abs(a.z), abs(a.w));
}

template <class T>
FORCEINLINE Vec4<T> rcp(const Vec4<T>& a)
{
    return Vec4<T>(rcp(a.x), rcp(a.y), rcp(a.z), rcp(a.w));
}

template <class T>
FORCEINLINE Vec4<T> rcpSafe(const Vec4<T>& a)
{
    return Vec4<T>(rcpSafe(a.x), rcpSafe(a.y), rcpSafe(a.z), rcpSafe(a.w));
}

template <class T>
FORCEINLINE Vec4<T> sqrt(const Vec4<T>& a)
{
    return Vec4<T>(sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w));
}

template <class T>
FORCEINLINE Vec4<T> floor(const Vec4<T>& a)
{
    return Vec4<T>(floor(a.x), floor(a.y), floor(a.z), floor(a.w));
}

template <class T>
FORCEINLINE Vec4<T> min(const Vec4<T>& a, const Vec4<T>& b)
{
	return Vec4<T>(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

template <class T>
FORCEINLINE Vec4<T> max(const Vec4<T>& a, const Vec4<T>& b)
{
	return Vec4<T>(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

template <class T>
FORCEINLINE Vec4<T> clamp(const Vec4<T>& value, const T& minValue, const T& maxValue)
{
    return Vec4<T>(clamp(value.x, minValue, maxValue),
                   clamp(value.y, minValue, maxValue),
                   clamp(value.z, minValue, maxValue),
                   clamp(value.w, minValue, maxValue));
}

template <class T>
FORCEINLINE ToBoolT<Vec4<T>> isfinite(const Vec4<T>& a)
{
    return ToBoolT<Vec4<T>>(isfinite(a.x), isfinite(a.y), isfinite(a.z), isfinite(a.w));
}

// Vector math functions
// ---------------------

template <class T>
FORCEINLINE T dot(const Vec4<T>& a, const Vec4<T>& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template <class T>
FORCEINLINE T lengthSqr(const Vec4<T>& a)
{
    return dot(a, a);
}

template <class T>
FORCEINLINE T length(const Vec4<T>& a)
{
    return sqrt(dot(a, a));
}

template <class T>
FORCEINLINE Vec4<T> normalize(const Vec4<T>& a)
{
    return a * rsqrt(dot(a, a));
}

// Reduction functions
// -------------------

template <class T>
FORCEINLINE T reduceMin(const Vec4<T>& a)
{
    return min(min(a.x, a.y), min(a.z, a.w));
}

template <class T>
FORCEINLINE T reduceMax(const Vec4<T>& a)
{
    return max(max(a.x, a.y), max(a.z, a.w));
}

template <class T>
FORCEINLINE T reduceAdd(const Vec4<T>& a)
{
    return a.x + a.y + a.z + a.w;
}

// Test functions
// --------------

template <class T>
FORCEINLINE bool all(const Vec4<T>& a)
{
    return all(a.x) && all(a.y) && all(a.z) && all(a.w);
}

template <class T>
FORCEINLINE bool any(const Vec4<T>& a)
{
    return any(a.x) || any(a.y) || any(a.z) || any(a.w);
}

template <class T>
FORCEINLINE bool none(const Vec4<T>& a)
{
    return none(a.x) && none(a.y) && none(a.z) && none(a.w);
}

// Stream operators
// ----------------

template <class T>
FORCEINLINE std::ostream& operator <<(std::ostream& osm, const Vec4<T>& a)
{
    osm << a.x << "," << a.y << "," << a.z << "," << a.w;
    return osm;
}

template <class T>
FORCEINLINE std::istream& operator >>(std::istream& ism, Vec4<T>& a)
{
    ism >> a.x;
    if (ism.peek() == ',')
    {
        ism.ignore();
    }
    else
    {
        a.y = a.z = a.w = a.x;
        return ism;
    }

    ism >> a.y;
    if (ism.peek() == ',') ism.ignore();
    ism >> a.z;
    if (ism.peek() == ',') ism.ignore();
    ism >> a.w;
    return ism;
}

// SIMD functions
// --------------

template <class T, int N>
Vec4v<T,N> gather4(const var<bool,N>& mask, const T* ptr, const var<int,N>& idx)
{
    return Vec4v<T,N>(gather(mask, ptr,   idx),
                      gather(mask, ptr+1, idx),
                      gather(mask, ptr+2, idx),
                      gather(mask, ptr+3, idx));
}

} // namespace prt
