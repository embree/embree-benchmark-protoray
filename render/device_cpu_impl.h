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

#include "sys/ref.h"
#include "math/random.h"
#include "camera/camera.h"
#include "image/frame_buffer.h"
#include "renderer.h"
#include "scene.h"
#include "device.h"

namespace prt {

class DeviceCpuImpl
{
private:
    ref<Scene> scene;
    ref<Renderer> renderer;
    ref<Camera> camera;
    ref<FrameBuffer> frameBuffer;
    Random rng;

public:
    DeviceCpuImpl();

    static DeviceCpuImpl* create();
    static void destroy(DeviceCpuImpl* impl);

    // Scene
    static void initScene(DeviceCpuImpl* impl, const std::string& path, const Props& props);
    static Box3f getSceneBounds(DeviceCpuImpl* impl);

    // Renderer
    static Props initRenderer(DeviceCpuImpl* impl, const Props& props, const Props& stats);
    static Props render(DeviceCpuImpl* impl, const Props& stats);
    static Props queryRay(DeviceCpuImpl* impl, const Ray& ray);
    static Props queryPixel(DeviceCpuImpl* impl, int x, int y);

    // Camera
    static void initCamera(DeviceCpuImpl* impl, const Props& props);

    // FrameBuffer
    static void initFrame(DeviceCpuImpl* impl, const Vec2i& size, const Props& props);
    static void initToneMapper(DeviceCpuImpl* impl, const Props& props);
    static void clearFrame(DeviceCpuImpl* impl);
    static void blitFrame(DeviceCpuImpl* impl, Surface dest);
};

} // namespace prt


