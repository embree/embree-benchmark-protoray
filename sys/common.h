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

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <climits>
#include <memory>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <utility>

// x86 intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__INTEL_COMPILER)
#include <ia32intrin.h>
#endif

#ifndef DEBUG
// Release
#if defined(_MSC_VER)
#define FORCEINLINE __forceinline
#else
#define FORCEINLINE inline __attribute__((always_inline))
#endif
#else
// Debug
#define FORCEINLINE inline
#endif

#if defined(_MSC_VER)
#define LIKELY(expr) expr
#define UNLIKELY(expr) expr
#else
#define LIKELY(expr) __builtin_expect(expr,true)
#define UNLIKELY(expr) __builtin_expect(expr,false)
#endif

#define __STDC_LIMIT_MACROS

#ifdef PROFILE_MODE
#define PROFILE(expr) expr
#else
#define PROFILE(expr)
#endif

namespace prt {

typedef unsigned int uint;

using namespace ::std::placeholders;

// FIXME
using std::swap;

class Uncopyable
{
protected:
	Uncopyable() {}
	~Uncopyable() {}

private:
	Uncopyable(const Uncopyable&);
	const Uncopyable& operator =(const Uncopyable&);
};

template <class T>
using ResultOfT = typename std::result_of<T>::type;

enum Access
{
    accessRead,
    accessWrite,
    accessReadWrite,
    accessReadWriteDiscard
};

// Select function
// ---------------

FORCEINLINE int    select(bool p, int    a, int    b) { return p ? a : b; }
FORCEINLINE float  select(bool p, float  a, float  b) { return p ? a : b; }
FORCEINLINE double select(bool p, double a, double b) { return p ? a : b; }

FORCEINLINE void set(bool p, int&    a, int    b) { if (p) a = b; }
FORCEINLINE void set(bool p, float&  a, float  b) { if (p) a = b; }
FORCEINLINE void set(bool p, double& a, double b) { if (p) a = b; }

FORCEINLINE void set(bool p, int*    a, int    b) { if (p) *a = b; }
FORCEINLINE void set(bool p, float*  a, float  b) { if (p) *a = b; }
FORCEINLINE void set(bool p, double* a, double b) { if (p) *a = b; }

// Min/max functions
// -----------------

template <class T>
FORCEINLINE T min(const T& a, const T& b)
{
    return (a < b) ? a : b;
}

template <class T>
FORCEINLINE T min(const T& a, const T& b, const T& c)
{
    return min(min(a, b), c);
}

template <class T>
FORCEINLINE T min(const T& a, const T& b, const T& c, const T& d)
{
    return min(min(a, b), min(c, d));
}

template <class T>
FORCEINLINE T max(const T& a, const T& b)
{
    return (a > b) ? a : b;
}

template <class T>
FORCEINLINE T max(const T& a, const T& b, const T& c)
{
    return max(max(a, b), c);
}

template <class T>
FORCEINLINE T max(const T& a, const T& b, const T& c, const T& d)
{
    return max(max(a, b), max(c, d));
}

// Test functions
// --------------

FORCEINLINE bool all(bool a) { return a; }
FORCEINLINE bool any(bool a) { return a; }
FORCEINLINE bool none(bool a) { return !a; }


template <class OutType, class InType>
FORCEINLINE OutType bitwise_cast(const InType& value)
{
	union
	{
		InType inValue;
		OutType outValue;
	}
	u;

	u.inValue = value;
	return u.outValue;
}

// Trait aliases
// -------------

#if defined(_LIBCPP_VERSION)
template <class T> using is_trivially_copy_constructible = std::is_trivially_copy_constructible<T>;
#else
template <class T> using is_trivially_copy_constructible = std::has_trivial_copy_constructor<T>;
#endif

#if __GNUC__ == 4 && __GNUC_MINOR__ < 8 && __GLIBCXX__ < 20140000
template <class T> using is_trivially_destructible = std::has_trivial_destructor<T>;
#else
template <class T> using is_trivially_destructible = std::is_trivially_destructible<T>;
#endif

} // namespace prt
