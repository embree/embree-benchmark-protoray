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

#include "sys/props.h"
#include "camera/camera.h"
#include "image/frame_buffer.h"
#include "scene.h"

namespace prt {

class Renderer
{
public:
    virtual ~Renderer() {}

    virtual void render(const Camera* camera, FrameBuffer* frameBuffer, Props& stats) = 0;
    virtual Props queryPixel(const Camera* camera, int x, int y) { return Props(); }
};

} // namespace prt
