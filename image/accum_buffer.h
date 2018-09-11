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

#include "sys/memory.h"
#include "sys/ref.h"
#include "sys/tasking.h"
#include "math/vec2.h"
#include "math/vec4.h"
#include "math/simd.h"
#include "pixel.h"
#include "color.h"
#include "surface.h"
#include "tone_mapper.h"

namespace prt {

enum AccumFlag
{
    accumFlagNone = 0,
};

template <int N>
class AccumBuffer : Uncopyable
{
protected:
    float* sum[N];
    float* weight;
    Vec2i size;

    void init(const Vec2i& size, int flags)
    {
        this->size = size;

        for (int i = 0; i < N; ++i)
            sum[i] = (float*)alignedAlloc(size.x * size.y * sizeof(float));

        weight = (float*)alignedAlloc(size.x * size.y * sizeof(float));

        clear();
    }

public:
    AccumBuffer()
    {
        for (int i = 0; i < N; ++i)
        {
            sum[i] = nullptr;
        }
        weight = nullptr;
    }

    AccumBuffer(const Vec2i& size, int flags)
    {
        init(size, flags);
    }

    ~AccumBuffer()
    {
        free();
    }

    void alloc(const Vec2i& size, int flags = 0)
    {
        free();
        init(size, flags);
    }

    void free()
    {
        for (int i = 0; i < N; ++i)
        {
            alignedFree(sum[i]);
        }
        alignedFree(weight);
    }

    FORCEINLINE Vec2i getSize() const
    {
        return size;
    }

    FORCEINLINE operator bool() const
    {
        return sum[0] != nullptr;
    }

    void clear()
    {
        if (sum[0] == nullptr)
            return;

        for (int i = 0; i < N; ++i)
        {
            tbb::parallel_for(tbb::blocked_range<int>(0, size.x*size.y), [&](const tbb::blocked_range<int>& r)
            {
                for (int j = r.begin(); j != r.end(); ++j)
                    sum[i][j] = 0.f;
            });
        }

        tbb::parallel_for(tbb::blocked_range<int>(0, size.x*size.y), [&](const tbb::blocked_range<int>& r)
        {
            for (int j = r.begin(); j != r.end(); ++j)
                weight[j] = 0.f;
        });
    }
};

class AccumBuffer3f : public AccumBuffer<3>
{
public:
    AccumBuffer3f() {}
    AccumBuffer3f(const Vec2i& size, int flags = 0) : AccumBuffer(size, flags) {}

    FORCEINLINE void add(int index, const Vec3f& c)
    {
        float x = sum[0][index] + c[0];
        float y = sum[1][index] + c[1];
        float z = sum[2][index] + c[2];
        float w = weight[index] + 1.f;

        sum[0][index] = x;
        sum[1][index] = y;
        sum[2][index] = z;
        weight[index] = w;
    }

    FORCEINLINE void add(const Vec2i& p, const Vec3f& c)
    {
        int index = p.y * size.x + p.x;
        add(index, c);
    }

    FORCEINLINE void add(vint index, const Vec3vf& c)
    {
        vfloat x = gather(sum[0], index) + c[0];
        vfloat y = gather(sum[1], index) + c[1];
        vfloat z = gather(sum[2], index) + c[2];
        vfloat w = gather(weight, index) + 1.f;

        scatter(sum[0], index, x);
        scatter(sum[1], index, y);
        scatter(sum[2], index, z);
        scatter(weight, index, w);
    }

    FORCEINLINE void add(vbool m, vint index, const Vec3vf& c)
    {
        vfloat x = gather(m, sum[0], index) + c[0];
        vfloat y = gather(m, sum[1], index) + c[1];
        vfloat z = gather(m, sum[2], index) + c[2];
        vfloat w = gather(m, weight, index) + 1.f;

        scatter(m, sum[0], index, x);
        scatter(m, sum[1], index, y);
        scatter(m, sum[2], index, z);
        scatter(m, weight, index, w);
    }

    // Serial version for duplicate indices
    FORCEINLINE void addSerial(vint index, const Vec3vf& c)
    {
        #pragma unroll
        for (int i = 0; i < simdSize; ++i)
        {
            int indexI = index[i];

            float x = sum[0][indexI] + c[0][i];
            float y = sum[1][indexI] + c[1][i];
            float z = sum[2][indexI] + c[2][i];
            float w = weight[indexI] + 1.f;

            sum[0][indexI] = x;
            sum[1][indexI] = y;
            sum[2][indexI] = z;
            weight[indexI] = w;
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

                float x = sum[0][indexI] + c[0][i];
                float y = sum[1][indexI] + c[1][i];
                float z = sum[2][indexI] + c[2][i];
                float w = weight[indexI]   + 1.f;

                sum[0][indexI] = x;
                sum[1][indexI] = y;
                sum[2][indexI] = z;
                weight[indexI] = w;
            }
        }
    }

    FORCEINLINE void add(const Vec2vi& p, const Vec3vf& c)
    {
        add(p.y * size.x + p.x, c);
    }

    FORCEINLINE void set(int index, const Vec3f& c)
    {
        sum[0][index] = c[0];
        sum[1][index] = c[1];
        sum[2][index] = c[2];
        weight[index] = 1.f;
    }

    FORCEINLINE void set(const Vec2i& p, const Vec3f& c)
    {
        int index = p.y * size.x + p.x;
        set(index, c);
    }

    FORCEINLINE void set(vint index, const Vec3vf& c)
    {
        scatter(sum[0], index, c[0]);
        scatter(sum[1], index, c[1]);
        scatter(sum[2], index, c[2]);
        scatter(weight, index, 1.f);
    }

    FORCEINLINE void set(vbool m, vint index, const Vec3vf& c)
    {
        scatter(m, sum[0], index, c[0]);
        scatter(m, sum[1], index, c[1]);
        scatter(m, sum[2], index, c[2]);
        scatter(m, weight, index, 1.f);
    }

    FORCEINLINE Vec3vf get(int index) const
    {
        vfloat x = load(sum[0] + index);
        vfloat y = load(sum[1] + index);
        vfloat z = load(sum[2] + index);
        vfloat w = load(weight + index);

        return Vec3vf(x, y, z) * rcpSafe(w);
    }

    void blit(Vec3f* dest) const
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, size.x*size.y), [&](const tbb::blocked_range<int>& r)
        {
            for (int i = r.begin(); i != r.end(); ++i)
            {
                float x = sum[0][i];
                float y = sum[1][i];
                float z = sum[2][i];
                float w = weight[i];

                dest[i] = Vec3f(x, y, z) * rcpSafe(w);
            }
        });
    }

    void blitLdr(Surface& dest, ToneMapper* toneMapper = nullptr) const
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, size.y), [&](const tbb::blocked_range<int>& r)
        {
            for (int iy = r.begin(); iy != r.end(); ++iy)
            {
                int i = iy * size.x;
                int* outRow = dest.getRow(iy);

                for (int ix = 0; ix < size.x; ix += simdSize)
                {
                    vfloat x = load(sum[0] + i);
                    vfloat y = load(sum[1] + i);
                    vfloat z = load(sum[2] + i);
                    vfloat w = load(weight + i);

                    Vec3vf C = Vec3vf(x, y, z) * rcpSafe(w);

                    if (toneMapper)
                        C = toneMapper->get(C);

                    store(outRow + ix, encodeBgr8(C));
                    i += simdSize;
                }
            }
        });
    }
};

} // namespace prt
