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

#include <float.h>
#include "sys/common.cuh"

namespace prt {

const float pi = 3.14159265358979323846f;

CUDA_DEV_FORCEINLINE float rcp(float a)
{
    return 1.0f / a;
}

CUDA_DEV_FORCEINLINE float3 rcpSafe(const float3& a)
{
    const float eps = 0x1p-80f;
    float3 r;
    r.x = 1.0f / (fabs(a.x) > eps ? a.x : copysign(eps, a.x));
    r.y = 1.0f / (fabs(a.y) > eps ? a.y : copysign(eps, a.y));
    r.z = 1.0f / (fabs(a.z) > eps ? a.z : copysign(eps, a.z));
    return r;
}

CUDA_DEV_FORCEINLINE int shl(int a, int b)
{
    return uint(a) << uint(b);
}

CUDA_DEV_FORCEINLINE int shr(int a, int b)
{
    return uint(a) >> uint(b);
}

CUDA_DEV_FORCEINLINE float asFloat(int a)
{
    return __int_as_float(a);
}

CUDA_DEV_FORCEINLINE int asInt(float a)
{
    return __float_as_int(a);
}

CUDA_DEV_FORCEINLINE float toFloatUnorm(unsigned int a)
{
    return __uint2float_rd(a) * 0x1.p-32f;
}

template <class T>
CUDA_DEV_FORCEINLINE T clamp(const T& value, const T& minValue, const T& maxValue)
{
    return min(max(value, minValue), maxValue);
}

CUDA_DEV_FORCEINLINE float2 operator +(const float2& a, const float2& b) { return make_float2(a.x + b.x, a.y + b.y); }
CUDA_DEV_FORCEINLINE float2 operator *(const float2& a, const float2& b) { return make_float2(a.x * b.x, a.y * b.y); }
CUDA_DEV_FORCEINLINE float2 operator *(float a, const float2& b) { return make_float2(a * b.x, a * b.y); }
CUDA_DEV_FORCEINLINE float2 operator *(const float2& a, float b) { return make_float2(a.x * b, a.y * b); }
CUDA_DEV_FORCEINLINE float2 operator /(const float2& a, const float2& b) { return make_float2(a.x / b.x, a.y / b.y); }

CUDA_DEV_FORCEINLINE float3 operator -(const float3& a) { return make_float3(-a.x, -a.y, -a.z); }
CUDA_DEV_FORCEINLINE float3 operator +(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
CUDA_DEV_FORCEINLINE float3 operator -(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
CUDA_DEV_FORCEINLINE float3 operator *(const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
CUDA_DEV_FORCEINLINE float3 operator *(float a, const float3& b) { return make_float3(a * b.x, a * b.y, a * b.z); }
CUDA_DEV_FORCEINLINE float3 operator *(const float3& a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }

CUDA_DEV_FORCEINLINE float3 abs(const float3& a) { return make_float3(fabs(a.x), fabs(a.y), fabs(a.z)); }

CUDA_DEV_FORCEINLINE float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

CUDA_DEV_FORCEINLINE float3 cross(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

CUDA_DEV_FORCEINLINE float3 normalize(const float3& a)
{
    return a * rsqrt(dot(a, a));
}

CUDA_DEV_FORCEINLINE float lengthSqr(const float3& a)
{
    return dot(a, a);
}

CUDA_DEV_FORCEINLINE float reduceMax(const float3& a)
{
    return max(max(a.x, a.y), a.z);
}

CUDA_DEV_FORCEINLINE float4 operator +(const float4& a, const float4& b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

CUDA_DEV_FORCEINLINE float min4(float a, float b, float c, float d)
{
    return fminf(fminf(fminf(a, b), c), d);
}

CUDA_DEV_FORCEINLINE float max4(float a, float b, float c, float d)
{
    return fmaxf(fmaxf(fmaxf(a, b), c), d);
}

CUDA_DEV_FORCEINLINE void shr64(unsigned int& low, unsigned int& high, int count = 1)
{
#if __CUDA_ARCH__ >= 350
    asm("shf.r.clamp.b32 %0, %0, %1, %2;" : "+r"(low) : "r"(high), "r"(count));
#else
    low = (low >> count) + (high << (32 - count));
#endif
    high >>= count;
}

CUDA_DEV_FORCEINLINE void shl64(unsigned int& low, unsigned int& high, int count)
{
#if __CUDA_ARCH__ >= 350
    asm("shf.l.clamp.b32 %0, %1, %0, %2;" : "+r"(high) : "r"(low), "r"(count));
#else
    high = (high << count) + (low >> (32 - count));
#endif
    low <<= count;
}

CUDA_DEV_FORCEINLINE void shl64(unsigned int& low, unsigned int& high)
{
    asm("add.cc.u32 %0, %0, %0; \n"
        "addc.u32   %1, %1, %1; \n"
        : "+r"(low), "+r"(high));
}

// Basis3f
// -------

struct Basis3f
{
    float3 U;
    float3 V;
    float3 N;

    CUDA_DEV_FORCEINLINE Basis3f() {}
    CUDA_DEV_FORCEINLINE Basis3f(const float3& U, const float3& V, const float3& N) : U(U), V(V), N(N) {}
};

CUDA_DEV_FORCEINLINE float3 operator *(const Basis3f& basis, const float3& a)
{
    return a.x * basis.U + a.y * basis.V + a.z * basis.N;
}

// N must be normalized!
CUDA_DEV_FORCEINLINE void makeBasis(float3& U, float3& V, const float3& N)
{
    float3 U0 = make_float3(0.0f, N.z, -N.y);
    float3 U1 = make_float3(-N.z, 0.0f, N.x);
    U = normalize(lengthSqr(U0) > lengthSqr(U1) ? U0 : U1);
    V = normalize(cross(N, U));
}

CUDA_DEV_FORCEINLINE Basis3f makeBasis(const float3& N)
{
    Basis3f basis;
    makeBasis(basis.U, basis.V, N);
    basis.N = N;
    return basis;
}

} // namespace prt
