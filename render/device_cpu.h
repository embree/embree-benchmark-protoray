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
#include "device.h"
#include "device_cpu_impl.h"

namespace prt {

class DeviceCpu : public Device
{
private:
    DeviceCpuImpl* impl;

public:
    DeviceCpu();
    ~DeviceCpu();

    std::string getInfo();

    // Scene
    void initScene(const std::string& path, const Props& props);
    Box3f getSceneBounds();

    // Renderer
    void initRenderer(const Props& props, Props& stats);
    void render(Props& stats);
    Props queryRay(const Ray& ray);
    Props queryPixel(int x, int y);

    // Camera
    void initCamera(const Props& props);

    // Film
    void initFrame(const Vec2i& size, const Props& props);
    void initToneMapper(const Props& props);
    void clearFrame();
    void blitFrame(Surface& dest);
};

} // namespace prt
