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

#include <limits>
#include "sys/common.h"

namespace prt {

static struct Zero
{
    FORCEINLINE operator int()      const { return 0; }
    FORCEINLINE operator uint32_t() const { return 0; }
    FORCEINLINE operator int8_t()   const { return 0; }
    FORCEINLINE operator uint8_t()  const { return 0; }
    FORCEINLINE operator int16_t()  const { return 0; }
    FORCEINLINE operator uint16_t() const { return 0; }
    FORCEINLINE operator int64_t()  const { return 0; }
    FORCEINLINE operator uint64_t() const { return 0; }
    FORCEINLINE operator float()    const { return 0; }
    FORCEINLINE operator double()   const { return 0; }
} zero;

static struct One
{
    FORCEINLINE operator int()      const { return 1; }
    FORCEINLINE operator uint32_t() const { return 1; }
    FORCEINLINE operator int8_t()   const { return 1; }
    FORCEINLINE operator uint8_t()  const { return 1; }
    FORCEINLINE operator int16_t()  const { return 1; }
    FORCEINLINE operator uint16_t() const { return 1; }
    FORCEINLINE operator int64_t()  const { return 1; }
    FORCEINLINE operator uint64_t() const { return 1; }
    FORCEINLINE operator float()    const { return 1; }
    FORCEINLINE operator double()   const { return 1; }
} one;

static struct PosMax
{
    FORCEINLINE operator float() const { return std::numeric_limits<float>::max(); }
    FORCEINLINE operator double() const { return std::numeric_limits<double>::max(); }
} posMax;

static struct NegMax
{
    FORCEINLINE operator float() const { return -std::numeric_limits<float>::max(); }
    FORCEINLINE operator double() const { return -std::numeric_limits<double>::max(); }
} negMax;

static struct PosInf
{
    FORCEINLINE operator float() const { return std::numeric_limits<float>::infinity(); }
    FORCEINLINE operator double() const { return std::numeric_limits<double>::infinity(); }
} posInf;

static struct NegInf
{
    FORCEINLINE operator float() const { return -std::numeric_limits<float>::infinity(); }
    FORCEINLINE operator double() const { return -std::numeric_limits<double>::infinity(); }
} negInf;

static struct Qnan
{
    FORCEINLINE operator float() const { return std::numeric_limits<float>::quiet_NaN(); }
    FORCEINLINE operator double() const { return std::numeric_limits<double>::quiet_NaN(); }
} qnan;

static struct Ulp
{
    FORCEINLINE operator float() const { return std::numeric_limits<float>::epsilon(); }
    FORCEINLINE operator double() const { return std::numeric_limits<double>::epsilon(); }
} ulp;

static struct Step {} step;
static struct Empty {} empty;

static struct Pi
{
    FORCEINLINE operator float() const { return 3.14159265358979323846f; }
    FORCEINLINE operator double() const { return 3.14159265358979323846; }
} pi;

} // namespace prt
