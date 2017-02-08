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

#include "pixel.h"
#include "frame_buffer_cuda.h"

namespace prt {

FrameBufferCuda::FrameBufferCuda(const Vec2i& size)
{
    this->size = size;

    colorBuffer.alloc(size.x * size.y * 4);
    accumBuffer.alloc(size.x * size.y * 16);

    clear();
}

const void* FrameBufferCuda::map()
{
    return colorBuffer.map(accessRead);
}

void FrameBufferCuda::unmap()
{
    colorBuffer.unmap();
}

void FrameBufferCuda::clear()
{
    cudaMemset(accumBuffer.getData(), 0, size.x * size.y * 16);
}

void FrameBufferCuda::update()
{
    getAccumBuffer().update(colorBuffer.getData());
}

} // namespace prt
