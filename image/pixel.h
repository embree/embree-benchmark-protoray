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

#include "sys/memory.h"
#include "math/vec2.h"
#include "math/vec3.h"
#include "math/vec4.h"

namespace prt {

enum PixelFormat
{
    pixelFormatBgr8,
    pixelFormatRgb32f
};

enum PixelCodec
{
    pixelCodecSrgb,
    pixelCodecLinear,
    pixelCodecSignedLinear
};

extern const float decodeSrgb8Table[256]; // 2.2 gamma

FORCEINLINE int encodeBgr8(const Vec3f& c)
{
    Vec3i ci = clamp(toInt(pow(c, 1.0f/2.2f) * 255.0f), 0, 255);
    return ci.z | (ci.y << 8) | (ci.x << 16);
}

FORCEINLINE vint encodeBgr8(const Vec3vf& c)
{
    Vec3vi ci = clamp(toInt(pow(c, vfloat(1.0f/2.2f)) * vfloat(255.0f)), vint(0), vint(255));
    return ci.z | (ci.y << 8) | (ci.x << 16);
}

// For linearly encoded data (e.g., bump maps)
FORCEINLINE int encodeBgr8Linear(const Vec3f& c)
{
    Vec3i ci = clamp(toInt(c * 255.0f), 0, 255);
    return ci.z | (ci.y << 8) | (ci.x << 16);
}

FORCEINLINE Vec3f decodeBgr8(const Vec4uc& c)
{
    return Vec3f(decodeSrgb8Table[c.z], decodeSrgb8Table[c.y], decodeSrgb8Table[c.x]);
}

FORCEINLINE Vec3f decodeBgr8(int c)
{
    uint32_t b = (uint32_t)c & 0xff;
    uint32_t g = ((uint32_t)c >> 8) & 0xff;
    uint32_t r = ((uint32_t)c >> 16) & 0xff;

    return Vec3f(decodeSrgb8Table[r], decodeSrgb8Table[g], decodeSrgb8Table[b]);
}

FORCEINLINE Vec3vf decodeBgr8(vint c)
{
    vint b = c & 0xff;
    vint g = (c >> 8) & 0xff;
    vint r = (c >> 16) & 0xff;

    return Vec3vf(gather(decodeSrgb8Table, r),
                  gather(decodeSrgb8Table, g),
                  gather(decodeSrgb8Table, b));
}

// For linearly encoded data (e.g., bump maps)
FORCEINLINE Vec3f decodeBgr8Linear(int c)
{
    uint32_t b = (uint32_t)c & 0xff;
    uint32_t g = ((uint32_t)c >> 8) & 0xff;
    uint32_t r = ((uint32_t)c >> 16) & 0xff;

    return Vec3f((float)r, (float)g, (float)b) * (1.0f/255.0f);
}

FORCEINLINE Vec3vf decodeBgr8Linear(vint c)
{
    vint b = c & 0xff;
    vint g = (c >> 8) & 0xff;
    vint r = (c >> 16) & 0xff;

    return toFloat(Vec3vi(r, g, b)) * vfloat(1.0f/255.0f);
}

// For normal maps
FORCEINLINE Vec3f decodeBgr8SignedLinear(int c)
{
    uint32_t b = (uint32_t)c & 0xff;
    uint32_t g = ((uint32_t)c >> 8) & 0xff;
    uint32_t r = ((uint32_t)c >> 16) & 0xff;

    return Vec3f((float)r, (float)g, (float)b) * (2.0f/255.0f) - 1.0f;
}

FORCEINLINE Vec3vf decodeBgr8SignedLinear(vint c)
{
    vint b = c & 0xff;
    vint g = (c >> 8) & 0xff;
    vint r = (c >> 16) & 0xff;

    return toFloat(Vec3vi(r, g, b)) * vfloat(2.0f/255.0f) - vfloat(1.0f);
}

// TODO: gamma
FORCEINLINE int encodeBgr8Fast(const Vec3f& c)
{
    Vec3i ci = clamp(toInt(c * 255.0f), 0, 255);
    return ci.z | (ci.y << 8) | (ci.x << 16);
}

// TODO: gamma
FORCEINLINE vint encodeBgr8Fast(const Vec3vf& c)
{
    Vec3vi ci = clamp(toInt(c * vfloat(255.0f)), vint(0), vint(255));
    return ci.z | (ci.y << 8) | (ci.x << 16);
}

// decodePixel function
// --------------------

template <PixelCodec codec = pixelCodecSrgb>
Vec3f decodePixel(int c);

template <>
FORCEINLINE Vec3f decodePixel<pixelCodecSrgb>(int c) { return decodeBgr8(c); }

template <>
FORCEINLINE Vec3f decodePixel<pixelCodecLinear>(int c) { return decodeBgr8Linear(c); }

template <>
FORCEINLINE Vec3f decodePixel<pixelCodecSignedLinear>(int c) { return decodeBgr8SignedLinear(c); }

// Codec is ignored for float formats
template <PixelCodec codec = pixelCodecSrgb>
FORCEINLINE Vec3f decodePixel(const Vec3f& c) { return c; }

template <PixelCodec codec = pixelCodecSrgb>
Vec3vf decodePixel(vint c);

template <>
FORCEINLINE Vec3vf decodePixel<pixelCodecSrgb>(vint c) { return decodeBgr8(c); }

template <>
FORCEINLINE Vec3vf decodePixel<pixelCodecLinear>(vint c) { return decodeBgr8Linear(c); }

template <>
FORCEINLINE Vec3vf decodePixel<pixelCodecSignedLinear>(vint c) { return decodeBgr8SignedLinear(c); }

// Codec is ignored for float formats
template <PixelCodec codec = pixelCodecSrgb>
FORCEINLINE Vec3vf decodePixel(const Vec3vf& c) { return c; }

} // namespace prt
