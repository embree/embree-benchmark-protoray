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

#include "sys/memory.h"
#include "frame_buffer.h"

namespace prt {

FrameBuffer::FrameBuffer(const Vec2i& size, int colorFlags)
    : color(size, colorFlags)
{
    invSize = rcp(toFloat(size));
}

void FrameBuffer::clear()
{
    color.clear();
}

void FrameBuffer::blitLdr(Surface& dest) const
{
    color.blitLdr(dest, toneMapper.get());
}

} // namespace prt
