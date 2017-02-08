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

#include "sys/ref.h"
#include "math/vec2.h"
#include "math/vec4.h"
#include "math/simd.h"
#include "pixel.h"
#include "surface.h"
#include "tone_mapper.h"

namespace prt {

class FrameBuffer : Uncopyable
{
private:
    float* accumBuffer[4]; // r, g, b, weight
    ref<ToneMapper> toneMapper;
    Vec2i size;
    Vec2f invSize;

public:
    FrameBuffer(const Vec2i& size);
    ~FrameBuffer();

    void clear();
    void update(Surface& surface);

    FORCEINLINE void add(const Vec2i& p, const Vec3f& c)
    {
        size_t index = p.y * size.x + p.x;

        float r = accumBuffer[0][index] + c[0];
        float g = accumBuffer[1][index] + c[1];
        float b = accumBuffer[2][index] + c[2];
        float w = accumBuffer[3][index] + 1.0f;

        accumBuffer[0][index] = r;
        accumBuffer[1][index] = g;
        accumBuffer[2][index] = b;
        accumBuffer[3][index] = w;
    }

    FORCEINLINE void add(vint index, const Vec3vf& c)
    {
        vfloat r = gather(accumBuffer[0], index) + c[0];
        vfloat g = gather(accumBuffer[1], index) + c[1];
        vfloat b = gather(accumBuffer[2], index) + c[2];
        vfloat w = gather(accumBuffer[3], index) + 1.0f;

        scatter(accumBuffer[0], index, r);
        scatter(accumBuffer[1], index, g);
        scatter(accumBuffer[2], index, b);
        scatter(accumBuffer[3], index, w);
    }

    FORCEINLINE void add(vbool m, vint index, const Vec3vf& c)
    {
        vfloat r = gather(m, accumBuffer[0], index) + c[0];
        vfloat g = gather(m, accumBuffer[1], index) + c[1];
        vfloat b = gather(m, accumBuffer[2], index) + c[2];
        vfloat w = gather(m, accumBuffer[3], index) + 1.0f;

        scatter(m, accumBuffer[0], index, r);
        scatter(m, accumBuffer[1], index, g);
        scatter(m, accumBuffer[2], index, b);
        scatter(m, accumBuffer[3], index, w);
    }

    // Serial version for duplicate indices
    FORCEINLINE void addSerial(vint index, const Vec3vf& c)
    {
        #pragma unroll
        for (int i = 0; i < simdSize; ++i)
        {
            int indexI = index[i];

            float r = accumBuffer[0][indexI] + c[0][i];
            float g = accumBuffer[1][indexI] + c[1][i];
            float b = accumBuffer[2][indexI] + c[2][i];
            float w = accumBuffer[3][indexI] + 1.0f;

            accumBuffer[0][indexI] = r;
            accumBuffer[1][indexI] = g;
            accumBuffer[2][indexI] = b;
            accumBuffer[3][indexI] = w;
        }
    }

    FORCEINLINE void addSerial(vbool m, vint index, const Vec3vf& c)
    {
        int mInt = toIntMask(m);

        #pragma unroll
        for (int i = 0; i < simdSize; ++i)
        {
            if (mInt & (1 << i))
            {
                int indexI = index[i];

                float r = accumBuffer[0][indexI] + c[0][i];
                float g = accumBuffer[1][indexI] + c[1][i];
                float b = accumBuffer[2][indexI] + c[2][i];
                float w = accumBuffer[3][indexI] + 1.0f;

                accumBuffer[0][indexI] = r;
                accumBuffer[1][indexI] = g;
                accumBuffer[2][indexI] = b;
                accumBuffer[3][indexI] = w;
            }
        }
    }

    FORCEINLINE void add(const Vec2vi& p, const Vec3vf& c)
    {
        add(p.y * size.x + p.x, c);
    }

    FORCEINLINE void set(const Vec2i& p, const Vec3f& c)
    {
        size_t index = p.y * size.x + p.x;

        accumBuffer[0][index] = c[0];
        accumBuffer[1][index] = c[1];
        accumBuffer[2][index] = c[2];
        accumBuffer[3][index] = 1.0f;
    }

    FORCEINLINE void set(vint index, const Vec3vf& c)
    {
        scatter(accumBuffer[0], index, c[0]);
        scatter(accumBuffer[1], index, c[1]);
        scatter(accumBuffer[2], index, c[2]);
        scatter(accumBuffer[3], index, 1.0f);
    }

    FORCEINLINE void set(vbool m, vint index, const Vec3vf& c)
    {
        scatter(m, accumBuffer[0], index, c[0]);
        scatter(m, accumBuffer[1], index, c[1]);
        scatter(m, accumBuffer[2], index, c[2]);
        scatter(m, accumBuffer[3], index, 1.0f);
    }

    void readHdr(Vec3f* dest);

    FORCEINLINE Vec2i getSize() const { return size; }
    FORCEINLINE Vec2f getInvSize() const { return invSize; }

    void setToneMapper(const ref<ToneMapper>& tm) { toneMapper = tm; }
};

} // namespace prt
