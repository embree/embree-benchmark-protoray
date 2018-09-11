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

#include "sys/buffer_cuda.h"
#include "math/vec2.h"
#include "math/vec4.h"
#include "accum_buffer_cuda.h"

namespace prt {

class FrameBufferCuda
{
private:
    BufferCuda colorBuffer;
    BufferCuda accumBuffer;
    Vec2i size;

public:
    FrameBufferCuda(const Vec2i& size);

    const void* map();
    void unmap();
    void clear();
    void update();

    FORCEINLINE Vec2i getSize() const { return size; }

    FORCEINLINE AccumBufferCuda getAccumBuffer()
    {
        AccumBufferCuda res;
        res.data = (float4*)accumBuffer.getData();
        res.size = make_int2(size.x, size.y);
        return res;
    }
};

} // namespace prt
