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

#include <cfloat>
#include <cmath>
#include "sys/common.h"

namespace prt {

// Rounding mode
enum Round
{
    roundNearest = 0,
    roundDown = 1,
    roundUp = 2,
    roundTruncate = 3,
    roundDefault = 4 // use MXCSR.RC
};

// Traits
// ------

template <class T>
struct ToFloat { typedef float Type; };

template <class T>
struct ToInt { typedef int Type; };

template <class T>
struct ToDouble { typedef double Type; };

template <class T>
struct ToBool { typedef bool Type; };

// Helper types
// ------------

template <class T>
using ToFloatT = typename ToFloat<T>::Type;

template <class T>
using ToIntT = typename ToInt<T>::Type;

template <class T>
using ToDoubleT = typename ToDouble<T>::Type;

template <class T>
using ToBoolT = typename ToBool<T>::Type;

} // namespace prt
