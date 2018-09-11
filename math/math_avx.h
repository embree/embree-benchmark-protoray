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

#include <immintrin.h>
#include "math/math_common.h"

namespace prt {

FORCEINLINE float rcp(float x)
{
    __m128 r = _mm_rcp_ss(_mm_set_ss(x));
    return _mm_cvtss_f32(_mm_sub_ss(_mm_add_ss(r, r), _mm_mul_ss(_mm_mul_ss(r, r), _mm_set_ss(x))));
}

FORCEINLINE float rsqrt(float x)
{
    __m128 r = _mm_rsqrt_ss(_mm_set_ss(x));
    return _mm_cvtss_f32(_mm_add_ss(_mm_mul_ss(_mm_set_ss(1.5f), r), _mm_mul_ss(_mm_mul_ss(_mm_mul_ss(_mm_set_ss(x), _mm_set_ss(-0.5f)), r), _mm_mul_ss(r, r))));
}

FORCEINLINE float floor(float x)
{
	return _mm_cvtss_f32(_mm_floor_ss(_mm_set_ss(x), _mm_set_ss(x)));
}

FORCEINLINE float ceil(float x)
{
    return _mm_cvtss_f32(_mm_ceil_ss(_mm_set_ss(x), _mm_set_ss(x)));
}

FORCEINLINE float round(float x)
{
    return _mm_cvtss_f32(_mm_round_ss(_mm_set_ss(x), _mm_set_ss(x), _MM_FROUND_TO_NEAREST_INT));
}

FORCEINLINE unsigned int bitCount(uint32_t x)
{
    return _mm_popcnt_u32(x);
}

// If the value is 0, the result is undefined!
FORCEINLINE unsigned int bitScan(uint32_t x)
{
#if defined(__AVX2__)
    return _tzcnt_u32(x);
#elif defined(_WIN32)
    unsigned long index = 0;
    _BitScanForward(&index, x);
    return index;
#else
    unsigned int index = 0; // remove initialization?
    asm("bsf %1,%0" : "=r"(index) : "r"(x));
    return index;
#endif
}

// Starts scanning from prevPos+1
// If not found, the result is 32
FORCEINLINE unsigned int bitScan(uint32_t x, unsigned int prevPos)
{
    unsigned int startPos = prevPos + 1;
    uint32_t xt = x >> startPos;
    if (xt == 0) return 32;
    return bitScan(xt) + startPos;
}

#ifdef NDEBUG // FIXME: GCC workaround for "impossible constraint in asm"
FORCEINLINE void shiftRight128(uint64_t& low, uint64_t& high, int count)
{
    asm("shrd %2,%1,%0" : "=r"(low) : "r"(high), "J"(count), "0"(low) : "flags");
    high >>= count;
}

FORCEINLINE void shiftLeft128(uint64_t& low, uint64_t& high, int count)
{
    asm("shld %2,%1,%0" : "=r"(high) : "r"(low), "J"(count), "0"(high) : "flags");
    low <<= count;
}
#else
FORCEINLINE void shiftRight128(uint64_t& low, uint64_t& high, int count)
{
    low = (low >> count) | (high << (64 - count));
    high >>= count;
}

FORCEINLINE void shiftLeft128(uint64_t& low, uint64_t& high, int count)
{
    high = (high << count) | (low >> (64 - count));
    low <<= count;
}
#endif

} // namespace prt
