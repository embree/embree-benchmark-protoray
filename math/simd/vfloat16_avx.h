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
#include "vbool16_avx.h"
#include "vint16_avx.h"

namespace prt {

// vfloat16
template <>
struct var<float,16>
{
    union
    {
        __m128 m[4];
        float v[16];
    };

    FORCEINLINE var() {}

    FORCEINLINE var(const vfloat16& a)
    {
        m[0] = a.m[0];
        m[1] = a.m[1];
        m[2] = a.m[2];
        m[3] = a.m[3];
    }

    FORCEINLINE var(__m128 a0, __m128 a1, __m128 a2, __m128 a3)
    {
        init(a0, a1, a2, a3);
    }

    FORCEINLINE var(__m128 a)
    {
        init(a);
    }

    FORCEINLINE var(float a)
	{
        init(a);
	}

    FORCEINLINE var(float a0, float a1, float a2, float a3)
    {
        init(a0, a1, a2, a3);
    }

    FORCEINLINE var(float a0, float a1, float a2,  float a3,  float a4,  float a5,  float a6,  float a7,
                   float a8, float a9, float a10, float a11, float a12, float a13, float a14, float a15)
    {
        init(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15);
    }

    FORCEINLINE var(Zero)   { init(_mm_setzero_ps()); }
    FORCEINLINE var(One)    { init(1.0f); }
    FORCEINLINE var(PosMax) { init(posMax); }
    FORCEINLINE var(NegMax) { init(negMax); }
    FORCEINLINE var(PosInf) { init(posInf); }
    FORCEINLINE var(NegInf) { init(negInf); }
    FORCEINLINE var(Qnan)   { init(qnan); }
    FORCEINLINE var(Step)   { init(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); }

    FORCEINLINE vfloat16& operator =(const vfloat16& a)
    {
        m[0] = a.m[0];
        m[1] = a.m[1];
        m[2] = a.m[2];
        m[3] = a.m[3];
        return *this;
    }

    FORCEINLINE const float& operator [](size_t i) const
	{
		assert(i < 16);
        return v[i];
	}

    FORCEINLINE float& operator [](size_t i)
	{
		assert(i < 16);
        return v[i];
	}

    FORCEINLINE const int& getInt(size_t i) const
    {
        assert(i < 16);
        return ((const int*)this)[i];
    }

    FORCEINLINE int& getInt(size_t i)
    {
        assert(i < 16);
        return ((int*)this)[i];
    }

    // Swizzle functions
    // -----------------

    FORCEINLINE vfloat16 yxwz() const { return swizzle<1,0,3,2>(); }
    FORCEINLINE vfloat16 zwxy() const { return swizzle<2,3,0,1>(); }
    FORCEINLINE vfloat16 yzxw() const { return swizzle<1,2,0,3>(); }
    FORCEINLINE vfloat16 xxxx() const { return swizzle<0,0,0,0>(); }
    FORCEINLINE vfloat16 yyyy() const { return swizzle<1,1,1,1>(); }
    FORCEINLINE vfloat16 zzzz() const { return swizzle<2,2,2,2>(); }
    FORCEINLINE vfloat16 wwww() const { return swizzle<3,3,3,3>(); }

private:
    // Helper functions
    // ----------------

    FORCEINLINE void init(__m128 a)
    {
        for (int i = 0; i < 4; ++i)
            m[i] = a;
    }

    FORCEINLINE void init(__m128 a0, __m128 a1, __m128 a2, __m128 a3)
    {
        m[0] = a0;
        m[1] = a1;
        m[2] = a2;
        m[3] = a3;
    }

    FORCEINLINE void init(float a)
    {
        init(_mm_set1_ps(a));
    }

    FORCEINLINE void init(float a0, float a1, float a2, float a3)
    {
        init(_mm_set_ps(a3, a2, a1, a0));
    }

    FORCEINLINE void init(float a0, float a1, float a2,  float a3,  float a4,  float a5,  float a6,  float a7,
                         float a8, float a9, float a10, float a11, float a12, float a13, float a14, float a15)
    {
        m[0] = _mm_set_ps(a3, a2, a1, a0);
        m[1] = _mm_set_ps(a7, a6, a5, a4);
        m[2] = _mm_set_ps(a11, a10, a9, a8);
        m[3] = _mm_set_ps(a15, a14, a13, a12);
    }

    template <int i0, int i1, int i2, int i3>
    FORCEINLINE vfloat16 swizzle() const
    {
        vfloat16 r;
        for (int i = 0; i < 4; ++i)
            r.m[i] = _mm_shuffle_ps(m[i], m[i], _MM_SHUFFLE(i3, i2, i1, i0));
        return r;
    }
};

// Load functions
// --------------

template <>
FORCEINLINE vfloat16 load(const float* ptr)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_load_ps(ptr + i*4);
    return r;
}

template <>
FORCEINLINE vfloat16 load(const vbool16& mask, const float* ptr)
{
    vfloat16 r;
    for (int i = 0; i < 16; ++i)
        if (mask.m & (1 << i)) r[i] = ptr[i];
    return r;
}

template <>
FORCEINLINE vfloat16 uload(const float* ptr)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_loadu_ps(ptr + i*4);
    return r;
}

template <>
FORCEINLINE vfloat16 load4(const float* ptr)
{
    return vfloat16(ptr[0], ptr[1], ptr[2], ptr[3]);
}

FORCEINLINE vfloat16 uexpand(const vbool16& mask, const float* ptr)
{
    vfloat16 r;
    int offset = 0;
    for (int i = 0; i < 16; ++i)
        if (mask.m & (1 << i)) r[i] = ptr[offset++];
    return r;
}

FORCEINLINE vfloat16 gather(const float* ptr, const vint16& idx)
{
    vfloat16 r;
    for (int i = 0; i < 16; ++i)
        r[i] = ptr[idx[i]];
    return r;
}

FORCEINLINE vfloat16 gather(const vbool16& mask, const float* ptr, const vint16& idx)
{
    vfloat16 r;
    for (int i = 0; i < 16; ++i)
        if (mask.m & (1 << i)) r[i] = ptr[idx[i]];
    return r;
}

FORCEINLINE void prefetchGatherL1(const float* ptr, const vint16& idx) {}
FORCEINLINE void prefetchGatherL2(const float* ptr, const vint16& idx) {}
FORCEINLINE void prefetchGatherL1(const vbool16& mask, const float* ptr, const vint16& idx) {}
FORCEINLINE void prefetchGatherL2(const vbool16& mask, const float* ptr, const vint16& idx) {}

// Store functions
// ---------------

FORCEINLINE void store(float* ptr, const vfloat16& a)
{
    for (int i = 0; i < 4; ++i)
        _mm_store_ps(ptr + i*4, a.m[i]);
}

FORCEINLINE void store(const vbool16& mask, float* ptr, const vfloat16& a)
{
    for (int i = 0; i < 16; ++i)
        if (mask.m & (1 << i)) ptr[i] = a[i];
}

FORCEINLINE void ustore(float* ptr, const vfloat16& a)
{
    for (int i = 0; i < 4; ++i)
        _mm_storeu_ps(ptr + i*4, a.m[i]);
}

FORCEINLINE void ucompress(const vbool16& mask, float* ptr, const vfloat16& a)
{
    int offset = 0;
    for (int i = 0; i < 16; ++i)
        if (mask.m & (1 << i)) ptr[offset++] = a[i];
}

FORCEINLINE void compress(const vbool16& mask, float* ptr, const vfloat16& a)
{
    ucompress(mask, ptr, a);
}

FORCEINLINE void scatter(float* ptr, const vint16& idx, const vfloat16& a)
{
    for (int i = 0; i < 16; ++i)
        ptr[idx[i]] = a[i];
}

FORCEINLINE void scatter(const vbool16& mask, float* ptr, const vint16& idx, const vfloat16& a)
{
    for (int i = 0; i < 16; ++i)
        if (mask.m & (1 << i)) ptr[idx[i]] = a[i];
}

// Conversion functions
// --------------------

FORCEINLINE float toScalar(const vfloat16& a)
{
    return _mm_cvtss_f32(a.m[0]);
}

FORCEINLINE vfloat16 toFloat(const vint16& a)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_cvtepi32_ps(a.m[i]);
    return r;
}

FORCEINLINE vfloat16 toFloatUnorm(const vint16& a)
{
    // TODO: vectorize
    vfloat16 r;
    for (int i = 0; i < 16; ++i)
        r[i] = float(uint32_t(a[i])) * 0x1.0p-32f;
    return r;
}

FORCEINLINE vfloat16 asFloat(const vint16& a)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_castsi128_ps(a.m[i]);
    return r;
}

FORCEINLINE vint16 toInt(const vfloat16& a)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_cvttps_epi32(a.m[i]);
    return r;
}

FORCEINLINE vint16 asInt(const vfloat16& a)
{
    vint16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_castps_si128(a.m[i]);
    return r;
}

// Select function
// ---------------

FORCEINLINE vfloat16 select(const vbool16& mask, const vfloat16& a, const vfloat16& b)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
    {
        __m128 blendMask = *(const __m128*)(&simdMaskTable4[(mask.m >> (i*4)) & 0xf]);
        r.m[i] = _mm_blendv_ps(b.m[i], a.m[i], blendMask);
    }
    return r;
}

// Shuffle functions
// -----------------

FORCEINLINE vfloat16 permute(const vfloat16& a, const vint16& idx)
{
    vfloat16 r;
    for (int i = 0; i < 16; ++i)
        r[i] = a[idx[i]];
    return r;
}

// Arithmetic operators
// --------------------

FORCEINLINE vfloat16 operator -(const vfloat16& a)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_sub_ps(_mm_setzero_ps(), a.m[i]);
    return r;
}

FORCEINLINE vfloat16 operator +(const vfloat16& a, const vfloat16& b)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_add_ps(a.m[i], b.m[i]);
    return r;
}

FORCEINLINE vfloat16 operator -(const vfloat16& a, const vfloat16& b)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_sub_ps(a.m[i], b.m[i]);
    return r;
}

FORCEINLINE vfloat16 operator *(const vfloat16& a, const vfloat16& b)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_mul_ps(a.m[i], b.m[i]);
    return r;
}

FORCEINLINE vfloat16 operator /(const vfloat16& a, const vfloat16& b)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
    {
        __m128 ri = _mm_rcp_ps(b.m[i]);
        ri = _mm_sub_ps(_mm_add_ps(ri, ri), _mm_mul_ps(_mm_mul_ps(ri, ri), b.m[i]));
        r.m[i] = _mm_mul_ps(a.m[i], ri);
    }
    return r;
}

// Bitwise operators
// -----------------

FORCEINLINE vfloat16 operator &(const vfloat16& a, const vfloat16& b)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_and_ps(a.m[i], b.m[i]);
    return r;
}

FORCEINLINE vfloat16 operator |(const vfloat16& a, const vfloat16& b)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_or_ps(a.m[i], b.m[i]);
    return r;
}

FORCEINLINE vfloat16 operator ^(const vfloat16& a, const vfloat16& b)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_xor_ps(a.m[i], b.m[i]);
    return r;
}

// Assignment operators
// --------------------

FORCEINLINE vfloat16& operator +=(vfloat16& a, const vfloat16& b) { return a = a + b; }
FORCEINLINE vfloat16& operator -=(vfloat16& a, const vfloat16& b) { return a = a - b; }
FORCEINLINE vfloat16& operator *=(vfloat16& a, const vfloat16& b) { return a = a * b; }
FORCEINLINE vfloat16& operator /=(vfloat16& a, const vfloat16& b) { return a = a / b; }
FORCEINLINE vfloat16& operator &=(vfloat16& a, const vfloat16& b) { return a = a & b; }
FORCEINLINE vfloat16& operator |=(vfloat16& a, const vfloat16& b) { return a = a | b; }
FORCEINLINE vfloat16& operator ^=(vfloat16& a, const vfloat16& b) { return a = a ^ b; }

// Compare operators
// -----------------

FORCEINLINE vbool16 operator ==(const vfloat16& a, const vfloat16& b)
{
    int mask = 0;
    for (int i = 0; i < 4; ++i)
        mask |= _mm_movemask_ps(_mm_cmpeq_ps(a.m[i], b.m[i])) << (i * 4);
    return mask;
}

FORCEINLINE vbool16 operator !=(const vfloat16& a, const vfloat16& b)
{
    int mask = 0;
    for (int i = 0; i < 4; ++i)
        mask |= _mm_movemask_ps(_mm_cmpneq_ps(a.m[i], b.m[i])) << (i * 4);
    return mask;
}

FORCEINLINE vbool16 operator <=(const vfloat16& a, const vfloat16& b)
{
    int mask = 0;
    for (int i = 0; i < 4; ++i)
        mask |= _mm_movemask_ps(_mm_cmple_ps(a.m[i], b.m[i])) << (i * 4);
    return mask;
}

FORCEINLINE vbool16 operator >=(const vfloat16& a, const vfloat16& b)
{
    int mask = 0;
    for (int i = 0; i < 4; ++i)
        mask |= _mm_movemask_ps(_mm_cmpge_ps(a.m[i], b.m[i])) << (i * 4);
    return mask;
}

FORCEINLINE vbool16 operator <(const vfloat16& a, const vfloat16& b)
{
    int mask = 0;
    for (int i = 0; i < 4; ++i)
        mask |= _mm_movemask_ps(_mm_cmplt_ps(a.m[i], b.m[i])) << (i * 4);
    return mask;
}

FORCEINLINE vbool16 operator >(const vfloat16& a, const vfloat16& b)
{
    int mask = 0;
    for (int i = 0; i < 4; ++i)
        mask |= _mm_movemask_ps(_mm_cmpgt_ps(a.m[i], b.m[i])) << (i * 4);
    return mask;
}

// Compare functions
// -----------------

FORCEINLINE vbool16 cmpEq(const vfloat16& a, const vfloat16& b) { return a == b; }
FORCEINLINE vbool16 cmpNe(const vfloat16& a, const vfloat16& b) { return a != b; }
FORCEINLINE vbool16 cmpLt(const vfloat16& a, const vfloat16& b) { return a <  b; }
FORCEINLINE vbool16 cmpLe(const vfloat16& a, const vfloat16& b) { return a <= b; }
FORCEINLINE vbool16 cmpGt(const vfloat16& a, const vfloat16& b) { return a >  b; }
FORCEINLINE vbool16 cmpGe(const vfloat16& a, const vfloat16& b) { return a >= b; }

FORCEINLINE vbool16 cmpEq(const vbool16& mask, const vfloat16& a, const vfloat16& b) { return (a == b) & mask; }
FORCEINLINE vbool16 cmpNe(const vbool16& mask, const vfloat16& a, const vfloat16& b) { return (a != b) & mask; }
FORCEINLINE vbool16 cmpLt(const vbool16& mask, const vfloat16& a, const vfloat16& b) { return (a <  b) & mask; }
FORCEINLINE vbool16 cmpLe(const vbool16& mask, const vfloat16& a, const vfloat16& b) { return (a <= b) & mask; }
FORCEINLINE vbool16 cmpGt(const vbool16& mask, const vfloat16& a, const vfloat16& b) { return (a >  b) & mask; }
FORCEINLINE vbool16 cmpGe(const vbool16& mask, const vfloat16& a, const vfloat16& b) { return (a >= b) & mask; }

// Math functions
// --------------

FORCEINLINE vfloat16 abs(const vfloat16& a)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_castsi128_ps(_mm_and_si128(_mm_set1_epi32(0x7fffffff), _mm_castps_si128(a.m[i])));
    return r;
}

FORCEINLINE vfloat16 rcp(const vfloat16& a)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
    {
        __m128 ri = _mm_rcp_ps(a.m[i]);
        r.m[i] = _mm_sub_ps(_mm_add_ps(ri, ri), _mm_mul_ps(_mm_mul_ps(ri, ri), a.m[i]));
    }
    return r;
}

FORCEINLINE vfloat16 sqrt(const vfloat16& a)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_sqrt_ps(a.m[i]);
    return r;
}

FORCEINLINE vfloat16 rsqrt(const vfloat16& a)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
    {
        __m128 ri = _mm_rsqrt_ps(a.m[i]);
        r.m[i] = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(1.5f), ri), _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(a.m[i], _mm_set1_ps(-0.5f)), ri), _mm_mul_ps(ri, ri)));
    }
    return r;
}

FORCEINLINE vfloat16 floor(const vfloat16& a)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_floor_ps(a.m[i]);
    return r;
}

FORCEINLINE vfloat16 ceil(const vfloat16& a)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_ceil_ps(a.m[i]);
    return r;
}

FORCEINLINE vfloat16 round(const vfloat16& a)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_round_ps(a.m[i], _MM_FROUND_TO_NEAREST_INT);
    return r;
}

FORCEINLINE vfloat16 min(const vfloat16& a, const vfloat16& b)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_min_ps(a.m[i], b.m[i]);
    return r;
}

FORCEINLINE vfloat16 max(const vfloat16& a, const vfloat16& b)
{
    vfloat16 r;
    for (int i = 0; i < 4; ++i)
        r.m[i] = _mm_max_ps(a.m[i], b.m[i]);
    return r;
}

FORCEINLINE vfloat16 vdot4(const vfloat16& a, const vfloat16& b)
{
    vfloat16 p = a * b;
    vfloat16 s = p + p.yxwz();
    return s + s.zwxy();
}

FORCEINLINE vfloat16 cross(const vfloat16& a, const vfloat16& b)
{
    vfloat16 p2 = b * a.yzxw();
    vfloat16 s = a * b.yzxw() - p2;
    return s.yzxw();
}

FORCEINLINE vfloat16 vreduceMin4(const vfloat16& a)
{
    vfloat16 t = min(a, a.yxwz());
    return min(t, t.zwxy());
}

FORCEINLINE vfloat16 vreduceMax4(const vfloat16& a)
{
    vfloat16 t = max(a, a.yxwz());
    return max(t, t.zwxy());
}

FORCEINLINE vfloat16 vreduceMinCross4(const vfloat16& a)
{
    return _mm_min_ps(_mm_min_ps(a.m[0], a.m[1]), _mm_min_ps(a.m[2], a.m[3]));
}

FORCEINLINE vfloat16 vreduceMaxCross4(const vfloat16& a)
{
    return _mm_max_ps(_mm_max_ps(a.m[0], a.m[1]), _mm_max_ps(a.m[2], a.m[3]));
}

FORCEINLINE vfloat16 vreduceMin(const vfloat16& a)
{
    return vreduceMin4(vreduceMinCross4(a));
}

FORCEINLINE vfloat16 vreduceMax(const vfloat16& a)
{
    return vreduceMax4(vreduceMaxCross4(a));
}

} // namespace prt
