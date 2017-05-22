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
#include "sys/cuda.h"
#include "sys/memory_cuda.h"
#include "geometry/triangle_mesh_cuda.h"
#include "core/intersector_stream_cuda.h"
#include "camera/thin_lens_camera_cuda.h"
#include "image/frame_buffer_cuda.h"
#include "device.h"
#include "renderer_cuda.h"

namespace prt {

class DeviceCuda : public Device
{
private:
    MemoryCuda meshIndices;
    MemoryCuda meshPositions;
    MemoryCuda meshNormals;
    MemoryCuda meshTexcoords;
    MemoryCuda bvh;
    TriangleMeshCuda mesh;
    ref<IntersectorStreamCuda> intersector;
    ref<FrameBufferCuda> frameBuffer;
    ref<RendererCuda> renderer;
    ThinLensCameraCuda camera;
    Box3f sceneBounds;
    std::string scenePath;

public:
    DeviceCuda();
    ~DeviceCuda();

    std::string getInfo();

    // Scene
    void initScene(const std::string& path, const Props& props);
    Box3f getSceneBounds();

    // Renderer
    void initRenderer(const Props& props, Props& stats);
    void render(Props& stats);
    Props queryPixel(int x, int y);

    // Camera
    void initCamera(const Props& props);

    // Film
    void initFrame(const Vec2i& size);
    void initToneMapper(const Props& props);
    void clearFrame();
    void updateFrame(Surface& surface);
    void readFrameHdr(Vec3f* dest);
};

} // namespace prt
