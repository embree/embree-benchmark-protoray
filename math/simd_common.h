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

#include "math.h"

namespace prt {

// Default SIMD size
#if (defined(__AVX512F__) || defined(__MIC__)) && !defined(FORCE_SIMD8)
#define SIMD_SIZE 16
const int simdSize = 16;
#else
#define SIMD_SIZE 8
const int simdSize = 8;
#endif

// Varying template
// ----------------

template <class T, int N = simdSize>
struct var;

// Functions
// ---------

template <int N = simdSize, class T>
var<T,N> load(const T* ptr);

template <int N = simdSize, class T>
var<T,N> load(const var<bool,N>& mask, const T* ptr);

template <int N = simdSize, class T>
var<T,N> load4(const T* ptr);

template <int N = simdSize, class T>
var<T,N> uload(const T* ptr);

// Typedefs
// --------

typedef var<float,simdSize> vfloat;
typedef var<float,4>        vfloat4;
typedef var<float,8>        vfloat8;
typedef var<float,16>       vfloat16;

typedef var<int,simdSize> vint;
typedef var<int,4>        vint4;
typedef var<int,8>        vint8;
typedef var<int,16>       vint16;

typedef var<bool,simdSize> vbool;
typedef var<bool,4>        vbool4;
typedef var<bool,8>        vbool8;
typedef var<bool,16>       vbool16;

// Traits
// ------

template <class T, int N>
struct ToFloat<var<T,N>> { typedef var<float,N> Type; };

template <class T, int N>
struct ToInt<var<T,N>> { typedef var<int,N> Type; };

template <class T, int N>
struct ToBool<var<T,N>> { typedef var<bool,N> Type; };

// Internals
// ---------

extern const uint32_t simdMaskTable4[16][4];
extern const uint32_t simdCompressTable8[256][8];
extern const uint32_t simdExpandTable8[256][8];

} // namespace prt
