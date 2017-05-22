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

#include "sys/memory.h"
#include "frame_buffer.h"

namespace prt {

FrameBuffer::FrameBuffer(const Vec2i& size)
{
    this->size = size;
    invSize = rcp(toFloat(size));

    for (int i = 0; i < 4; ++i)
        accumBuffer[i] = (float*)alignedAlloc(size.x * size.y * sizeof(float));

    clear();
}

FrameBuffer::~FrameBuffer()
{
    for (int i = 0; i < 4; ++i)
        alignedFree(accumBuffer[i]);
}

void FrameBuffer::clear()
{
    for (int i = 0; i < 4; ++i)
    {
        #pragma omp parallel for
        for (int j = 0; j < size.x*size.y; ++j)
            accumBuffer[i][j] = 0.0f;
    }
}

void FrameBuffer::update(Surface& surface)
{
    #pragma omp parallel for
    for (int y = 0; y < size.y; ++y)
    {
        int i = y * size.x;
        int* outRow = surface.getRow(y);

        for (int x = 0; x < size.x; x += simdSize)
        {
            vfloat r = load(accumBuffer[0] + i);
            vfloat g = load(accumBuffer[1] + i);
            vfloat b = load(accumBuffer[2] + i);
            vfloat w = load(accumBuffer[3] + i);

            Vec3vf d = Vec3vf(r, g, b) * rcp(w);
            if (toneMapper) d = toneMapper->get(d);
            store(outRow + x, encodeBgr8(d));
            i += simdSize;
        }
    }
}

void FrameBuffer::readHdr(Vec3f* dest)
{
    #pragma omp parallel for
    for (int i = 0; i < size.x*size.y; ++i)
    {
        float r = accumBuffer[0][i];
        float g = accumBuffer[1][i];
        float b = accumBuffer[2][i];
        float w = accumBuffer[3][i];

        dest[i] = Vec3f(r, g, b) * rcp(w);
    }
}

} // namespace prt
