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
#include "sys/props.h"
#include "math/vec2.h"
#include "math/vec3.h"
#include "math/box3.h"
#include "image/surface.h"

namespace prt {

class Device
{
public:
    virtual ~Device() {}

    virtual std::string getInfo() = 0;

    // Scene
    virtual void initScene(const std::string& path, const Props& props) = 0;
    virtual Box3f getSceneBounds() = 0;

    // Renderer (initialize the scene first)
    virtual void initRenderer(const Props& props, Props& stats) = 0;
    virtual void render(Props& stats) = 0;
    virtual Props queryPixel(int x, int y) = 0;

    // Camera
    virtual void initCamera(const Props& props) = 0;

    // Framebuffer
    virtual void initFrame(const Vec2i& size) = 0;
    virtual void initToneMapper(const Props& props) = 0;
    virtual void clearFrame() = 0;
    virtual void updateFrame(Surface& surface) = 0;
    virtual void readFrameHdr(Vec3f* dest) = 0;
};

} // namespace prt
