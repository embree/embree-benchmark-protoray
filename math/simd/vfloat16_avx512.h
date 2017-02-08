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
#include "vbool16_avx512.h"
#include "vint16_avx512.h"

namespace prt {

// vfloat16
template <>
struct var<float,16>
{
    union
    {
        __m512 m;
        float v[16];
    };

    FORCEINLINE var() {}

    FORCEINLINE var(const vfloat16& a) : m(a.m) {}
    FORCEINLINE var(__m512 a) : m(a) {}
    FORCEINLINE var(float a) : m(_mm512_set1_ps(a)) {}
    FORCEINLINE var(float a0, float a1, float a2, float a3) : m(_mm512_set4_ps(a3, a2, a1, a0)) {}

    FORCEINLINE var(float a0, float a1, float a2,  float a3,  float a4,  float a5,  float a6,  float a7,
                   float a8, float a9, float a10, float a11, float a12, float a13, float a14, float a15)
        : m(_mm512_set_ps(a15, a14, a13, a12, a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, a0)) {}

    FORCEINLINE var(Zero)   : m(_mm512_setzero_ps()) {}
    FORCEINLINE var(One)    : m(_mm512_set1_ps(1.0f)) {}
    FORCEINLINE var(PosMax) : m(_mm512_set1_ps(posMax)) {}
    FORCEINLINE var(NegMax) : m(_mm512_set1_ps(negMax)) {}
    FORCEINLINE var(PosInf) : m(_mm512_set1_ps(posInf)) {}
    FORCEINLINE var(NegInf) : m(_mm512_set1_ps(negInf)) {}
    FORCEINLINE var(Qnan)   : m(_mm512_set1_ps(qnan)) {}
    FORCEINLINE var(Step)   : m(_mm512_set_ps(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)) {}

    FORCEINLINE vfloat16& operator =(const vfloat16& b)
    {
        m = b.m;
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
    template <int i0, int i1, int i2, int i3>
    FORCEINLINE vfloat16 swizzle() const
    {
        return _mm512_shuffle_ps(m, m, _MM_SHUFFLE(i3, i2, i1, i0));
    }
};

// Load functions
// --------------

template <>
FORCEINLINE vfloat16 load(const float* ptr)
{
    return _mm512_load_ps(ptr);
}

template <>
FORCEINLINE vfloat16 load(const vbool16& mask, const float* ptr)
{
    return _mm512_mask_load_ps(_mm512_undefined_ps(), mask.m, ptr);
}

template <>
FORCEINLINE vfloat16 uload(const float* ptr)
{
    return _mm512_loadu_ps(ptr);
}

template <>
FORCEINLINE vfloat16 load4(const float* ptr)
{
    return _mm512_broadcast_f32x4(_mm_load_ps(ptr));
}

FORCEINLINE vfloat16 uexpand(const vbool16& mask, const float* ptr)
{
    return _mm512_maskz_expandloadu_ps(mask.m, ptr);
}

FORCEINLINE vfloat16 gather(const float* ptr, const vint16& idx)
{
    return _mm512_i32gather_ps(idx.m, ptr, 4);
}

FORCEINLINE vfloat16 gather(const vbool16& mask, const float* ptr, const vint16& idx)
{
    return _mm512_mask_i32gather_ps(_mm512_undefined_ps(), mask.m, idx.m, ptr, 4);
}

FORCEINLINE void prefetchGatherL1(const float* ptr, const vint16& idx)
{
#if defined(__AVX512PF__) || defined(__KNC__)
    _mm512_prefetch_i32gather_ps(idx.m, ptr, 4, _MM_HINT_T0);
#endif
}

FORCEINLINE void prefetchGatherL2(const float* ptr, const vint16& idx)
{
#if defined(__AVX512PF__) || defined(__KNC__)
    _mm512_prefetch_i32gather_ps(idx.m, ptr, 4, _MM_HINT_T1);
#endif
}

FORCEINLINE void prefetchGatherL1(const vbool16& mask, const float* ptr, const vint16& idx)
{
#if defined(__AVX512PF__) || defined(__KNC__)
    _mm512_mask_prefetch_i32gather_ps(idx.m, mask.m, ptr, 4, _MM_HINT_T0);
#endif
}

FORCEINLINE void prefetchGatherL2(const vbool16& mask, const float* ptr, const vint16& idx)
{
#if defined(__AVX512PF__) || defined(__KNC__)
    _mm512_mask_prefetch_i32gather_ps(idx.m, mask.m, ptr, 4, _MM_HINT_T1);
#endif
}

// Store functions
// ---------------

FORCEINLINE void store(float* ptr, const vfloat16& a)
{
    _mm512_store_ps(ptr, a.m);
}

FORCEINLINE void store(const vbool16& mask, float* ptr, const vfloat16& a)
{
    _mm512_mask_store_ps(ptr, mask.m, a.m);
}

FORCEINLINE void ustore(float* ptr, const vfloat16& a)
{
    _mm512_storeu_ps(ptr, a.m);
}

FORCEINLINE void ucompress(const vbool16& mask, float* ptr, const vfloat16& a)
{
    _mm512_mask_compressstoreu_ps(ptr, mask.m, a.m);
}

FORCEINLINE void compress(const vbool16& mask, float* ptr, const vfloat16& a)
{
    _mm512_mask_compressstoreu_ps(ptr, mask.m, a.m);
}

FORCEINLINE void scatter(float* ptr, const vint16& idx, const vfloat16& a)
{
    _mm512_i32scatter_ps(ptr, idx.m, a.m, 4);
}

FORCEINLINE void scatter(const vbool16& mask, float* ptr, const vint16& idx, const vfloat16& a)
{
    _mm512_mask_i32scatter_ps(ptr, mask.m, idx.m, a.m, 4);
}

// Conversion functions
// --------------------

FORCEINLINE float toScalar(const vfloat16& a)
{
    return _mm512_cvtss_f32(a.m);
}

FORCEINLINE vfloat16 toFloat(const vint16& a)
{
    return _mm512_cvt_roundepi32_ps(a.m, _MM_FROUND_CUR_DIRECTION);
}

FORCEINLINE vfloat16 toFloatUnorm(const vint16& a)
{
    return _mm512_mul_ps(_mm512_cvt_roundepu32_ps(a.m, _MM_FROUND_CUR_DIRECTION), _mm512_set1_ps(0x1.0p-32f));
}

FORCEINLINE vfloat16 asFloat(const vint16& a)
{
    return _mm512_castsi512_ps(a.m);
}

FORCEINLINE vint16 toInt(const vfloat16& a)
{
    return _mm512_cvt_roundps_epi32(a.m, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

FORCEINLINE vint16 asInt(const vfloat16& a)
{
    return _mm512_castps_si512(a.m);
}

// Select function
// ---------------

FORCEINLINE vfloat16 select(const vbool16& mask, const vfloat16& a, const vfloat16& b)
{
    return _mm512_mask_mov_ps(b.m, mask.m, a.m);
}

// Shuffle functions
// -----------------

FORCEINLINE vfloat16 permute(const vfloat16& a, const vint16& idx)
{
    return _mm512_castsi512_ps(_mm512_permutevar_epi32(idx.m, _mm512_castps_si512(a.m)));
}

// Arithmetic operators
// --------------------

FORCEINLINE vfloat16 operator -(const vfloat16& a)
{
    return _mm512_sub_ps(_mm512_setzero_ps(), a.m);
}

FORCEINLINE vfloat16 operator +(const vfloat16& a, const vfloat16& b)
{
    return _mm512_add_ps(a.m, b.m);
}

FORCEINLINE vfloat16 operator -(const vfloat16& a, const vfloat16& b)
{
    return _mm512_sub_ps(a.m, b.m);
}

FORCEINLINE vfloat16 operator *(const vfloat16& a, const vfloat16& b)
{
    return _mm512_mul_ps(a.m, b.m);
}

FORCEINLINE vfloat16 operator /(const vfloat16& a, const vfloat16& b)
{
    //return _mm512_div_ps(a.m, b.m);
#if defined(__AVX512ER__) || defined(__KNC__)
    return _mm512_mul_ps(a.m, _mm512_rcp28_ps(b.m));
#else
    __m512 r = _mm512_rcp14_ps(b.m);
    r = _mm512_sub_ps(_mm512_add_ps(r, r), _mm512_mul_ps(_mm512_mul_ps(r, r), b.m));
    return _mm512_mul_ps(a.m, r);
#endif
}

// Bitwise operators
// -----------------

FORCEINLINE vfloat16 operator &(const vfloat16& a, const vfloat16& b)
{
    return _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(a.m), _mm512_castps_si512(b.m)));
}

FORCEINLINE vfloat16 operator |(const vfloat16& a, const vfloat16& b)
{
    return _mm512_castsi512_ps(_mm512_or_epi32(_mm512_castps_si512(a.m), _mm512_castps_si512(b.m)));
}

FORCEINLINE vfloat16 operator ^(const vfloat16& a, const vfloat16& b)
{
    return _mm512_castsi512_ps(_mm512_xor_epi32(_mm512_castps_si512(a.m), _mm512_castps_si512(b.m)));
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
    return _mm512_cmp_ps_mask(a.m, b.m, _MM_CMPINT_EQ);
}

FORCEINLINE vbool16 operator !=(const vfloat16& a, const vfloat16& b)
{
    return _mm512_cmp_ps_mask(a.m, b.m, _MM_CMPINT_NE);
}

FORCEINLINE vbool16 operator <=(const vfloat16& a, const vfloat16& b)
{
    return _mm512_cmp_ps_mask(a.m, b.m, _MM_CMPINT_LE);
}

FORCEINLINE vbool16 operator >=(const vfloat16& a, const vfloat16& b)
{
    return _mm512_cmp_ps_mask(a.m, b.m, _MM_CMPINT_GE);
}

FORCEINLINE vbool16 operator <(const vfloat16& a, const vfloat16& b)
{
    return _mm512_cmp_ps_mask(a.m, b.m, _MM_CMPINT_LT);
}

FORCEINLINE vbool16 operator >(const vfloat16& a, const vfloat16& b)
{
    return _mm512_cmp_ps_mask(a.m, b.m, _MM_CMPINT_GT);
}

// Compare functions
// -----------------

FORCEINLINE vbool16 cmpEq(const vfloat16& a, const vfloat16& b) { return a == b; }
FORCEINLINE vbool16 cmpNe(const vfloat16& a, const vfloat16& b) { return a != b; }
FORCEINLINE vbool16 cmpLt(const vfloat16& a, const vfloat16& b) { return a <  b; }
FORCEINLINE vbool16 cmpLe(const vfloat16& a, const vfloat16& b) { return a <= b; }
FORCEINLINE vbool16 cmpGt(const vfloat16& a, const vfloat16& b) { return a >  b; }
FORCEINLINE vbool16 cmpGe(const vfloat16& a, const vfloat16& b) { return a >= b; }

FORCEINLINE vbool16 cmpEq(const vbool16& mask, const vfloat16& a, const vfloat16& b)
{
    return _mm512_mask_cmp_ps_mask(mask.m, a.m, b.m, _MM_CMPINT_EQ);
}

FORCEINLINE vbool16 cmpNe(const vbool16& mask, const vfloat16& a, const vfloat16& b)
{
    return _mm512_mask_cmp_ps_mask(mask.m, a.m, b.m, _MM_CMPINT_NE);
}

FORCEINLINE vbool16 cmpLt(const vbool16& mask, const vfloat16& a, const vfloat16& b)
{
    return _mm512_mask_cmp_ps_mask(mask.m, a.m, b.m, _MM_CMPINT_LT);
}

FORCEINLINE vbool16 cmpLe(const vbool16& mask, const vfloat16& a, const vfloat16& b)
{
    return _mm512_mask_cmp_ps_mask(mask.m, a.m, b.m, _MM_CMPINT_LE);
}

FORCEINLINE vbool16 cmpGt(const vbool16& mask, const vfloat16& a, const vfloat16& b)
{
    return _mm512_mask_cmp_ps_mask(mask.m, a.m, b.m, _MM_CMPINT_GT);
}

FORCEINLINE vbool16 cmpGe(const vbool16& mask, const vfloat16& a, const vfloat16& b)
{
    return _mm512_mask_cmp_ps_mask(mask.m, a.m, b.m, _MM_CMPINT_GE);
}

// Math functions
// --------------

FORCEINLINE vfloat16 abs(const vfloat16& a)
{
    return _mm512_abs_ps(a.m);
}

FORCEINLINE vfloat16 rcp(const vfloat16& a)
{
#if defined(__AVX512ER__) || defined(__KNC__)
    return _mm512_rcp28_ps(a.m);
#else
    __m512 r = _mm512_rcp14_ps(a.m);
    return _mm512_sub_ps(_mm512_add_ps(r, r), _mm512_mul_ps(_mm512_mul_ps(r, r), a.m));
#endif
}

FORCEINLINE vfloat16 sqrt(const vfloat16& a)
{
    return _mm512_sqrt_ps(a.m);
}

FORCEINLINE vfloat16 rsqrt(const vfloat16& a)
{
#if defined(__AVX512ER__) || defined(__KNC__)
    return _mm512_rsqrt28_ps(a.m);
#else
    __m512 r = _mm512_rsqrt14_ps(a.m);
    return _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(1.5f), r), _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(a.m, _mm512_set1_ps(-0.5f)), r), _mm512_mul_ps(r, r)));
#endif
}

FORCEINLINE vfloat16 floor(const vfloat16& a)
{
    return _mm512_floor_ps(a.m);
}

FORCEINLINE vfloat16 ceil(const vfloat16& a)
{
    return _mm512_ceil_ps(a.m);
}

FORCEINLINE vfloat16 round(const vfloat16& a)
{
    return _mm512_nearbyint_ps(a.m);
}

FORCEINLINE vfloat16 min(const vfloat16&a, const vfloat16& b)
{
    return _mm512_min_ps(a.m, b.m);
}

FORCEINLINE vfloat16 max(const vfloat16&a, const vfloat16& b)
{
    return _mm512_max_ps(a.m, b.m);
}

FORCEINLINE vfloat16 vdot4(const vfloat16& a, const vfloat16& b)
{
    __m512 p = _mm512_mul_ps(a.m, b.m);
    __m512 s = _mm512_add_ps(p, _mm512_swizzle_ps(p, _MM_SWIZ_REG_CDAB));
    return _mm512_add_ps(s, _mm512_swizzle_ps(s, _MM_SWIZ_REG_BADC));
}

FORCEINLINE vfloat16 cross(const vfloat16& a, const vfloat16& b)
{
    __m512 p2 = _mm512_mul_ps(b.m, _mm512_swizzle_ps(a.m, _MM_SWIZ_REG_DACB));
    __m512 s = _mm512_fmsub_ps(a.m, _mm512_swizzle_ps(b.m, _MM_SWIZ_REG_DACB), p2);
    return _mm512_swizzle_ps(s, _MM_SWIZ_REG_DACB);
}

FORCEINLINE vfloat16 vreduceMin4(const vfloat16& a)
{
    __m512 t = _mm512_min_ps(a.m, _mm512_swizzle_ps(a.m, _MM_SWIZ_REG_CDAB));
    return _mm512_min_ps(t, _mm512_swizzle_ps(t, _MM_SWIZ_REG_BADC));
}

FORCEINLINE vfloat16 vreduceMax4(const vfloat16& a)
{
    __m512 t = _mm512_max_ps(a.m, _mm512_swizzle_ps(a.m, _MM_SWIZ_REG_CDAB));
    return _mm512_max_ps(t, _mm512_swizzle_ps(t, _MM_SWIZ_REG_BADC));
}

FORCEINLINE vfloat16 vreduceMinCross4(const vfloat16& a)
{
    __m512 t = _mm512_min_ps(a.m, _mm512_permute4f128_ps(a.m, _MM_PERM_CDAB));
    return _mm512_min_ps(t, _mm512_permute4f128_ps(t, _MM_PERM_BADC));
}

FORCEINLINE vfloat16 vreduceMaxCross4(const vfloat16& a)
{
    __m512 t = _mm512_max_ps(a.m, _mm512_permute4f128_ps(a.m, _MM_PERM_CDAB));
    return _mm512_max_ps(t, _mm512_permute4f128_ps(t, _MM_PERM_BADC));
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
