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

#include "sys/logging.h"
#include "sys/filesystem.h"
#include "sys/blob.h"
#include "sys/timer_cuda.h"
#include "geometry/triangle_mesh.h"
#include "camera/pinhole_camera.h"
#include "camera/thin_lens_camera.h"
#include "accel/optix/optix_intersector_stream_cuda.h"
#include "primary_renderer_cuda.h"
#include "diffuse_renderer_cuda.h"
#include "device_cuda.h"

namespace prt {

DeviceCuda::DeviceCuda()
{
}

DeviceCuda::~DeviceCuda()
{
}

std::string DeviceCuda::getInfo()
{
    int dev;
    cudaGetDevice(&dev);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    return deviceProp.name;
}

void DeviceCuda::initRenderer(const Props& props, Props& stats)
{
    // Init the acceleration structure
    std::string accelType = props.get("accel", "optix");
    Log() << "Acceleration structure: " << accelType;

    if (accelType == "optix")
    {
        intersector = makeRef<OptixIntersectorStreamCuda>(mesh, props, stats);
    }
    else
    {
        throw std::invalid_argument("invalid acceleration structure type");
    }

    // Init the renderer
    std::string type = props.get("type");
    int maxDepth = props.get("maxDepth", 6);
    Vec2i imageSize = props.get<Vec2i>("imageSize");
    int pixelCount = imageSize.x * imageSize.y;

    std::string samplerType = props.get("sampler", "random");
    if (samplerType != "random")
        throw std::invalid_argument("invalid sampler type");

    if (type == "primary")
        renderer = makeRef<PrimaryRendererCuda>(mesh, intersector.get(), pixelCount);
    else if (type == "primaryFast")
        renderer = makeRef<PrimaryRendererCuda>(mesh, intersector.get(), pixelCount, true);
    else if (type == "diffuse")    
        renderer = makeRef<DiffuseRendererCuda>(mesh, intersector.get(), pixelCount, maxDepth);
    else if (type == "diffuseFast")
        renderer = makeRef<DiffuseRendererCuda>(mesh, intersector.get(), pixelCount, maxDepth, true);
    else
        throw std::invalid_argument("invalid renderer type");
}

void DeviceCuda::render(Props& stats)
{
    AccumBufferCuda accumBuffer = frameBuffer->getAccumBuffer();

    TimerCuda timer;
    timer.start();
    int rayCount = renderer->render(camera, accumBuffer);
    double totalTime = timer.stop();

    frameBuffer->update();
    double mray = (double)rayCount / 1000000.0 / totalTime;
    stats.set("mray", mray);
    stats.set("ray", rayCount);
    //stats.set("spp", 1);
}

Props DeviceCuda::queryPixel(int x, int y)
{
    return Props();
}

void DeviceCuda::initScene(const std::string& path, const Props& props)
{
    // Load the mesh
    ref<TriangleMesh> meshHost = makeRef<TriangleMesh>();
    loadBlob(path, *meshHost);

    int triangleCount = meshHost->indices.getSize();
    meshIndices.alloc(triangleCount * sizeof(Vec3i));
    cudaMemcpy(meshIndices.getData(), meshHost->indices.getData(), triangleCount * sizeof(Vec3i), cudaMemcpyHostToDevice);

    int vertexCount = meshHost->getVertexCount();
    Array<float> vertices(vertexCount*3);

    Vec3f* positions = (Vec3f*)vertices.getData();
    for (int i = 0; i < vertexCount; ++i)
        positions[i] = meshHost->getPosition(i);
    meshPositions.alloc(vertexCount * sizeof(Vec3f));
    cudaMemcpy(meshPositions.getData(), positions, vertexCount * sizeof(Vec3f), cudaMemcpyHostToDevice);

    if (meshHost->normals)
    {
        Vec3f* normals = (Vec3f*)vertices.getData();
        for (int i = 0; i < vertexCount; ++i)
            normals[i] = meshHost->getNormal(i);
        meshNormals.alloc(vertexCount * sizeof(Vec3f));
        cudaMemcpy(meshNormals.getData(), normals, vertexCount * sizeof(Vec3f), cudaMemcpyHostToDevice);
    }

    if (meshHost->texcoords)
    {
        Vec2f* texcoords = (Vec2f*)vertices.getData();
        for (int i = 0; i < vertexCount; ++i)
            texcoords[i] = meshHost->getTexcoord(i);
        meshTexcoords.alloc(vertexCount * sizeof(Vec2f));
        cudaMemcpy(meshTexcoords.getData(), texcoords, vertexCount * sizeof(Vec2f), cudaMemcpyHostToDevice);
    }

    mesh.indices = (int3*)meshIndices.getData();
    mesh.positions = (float3*)meshPositions.getData();
    mesh.normals = (float3*)meshNormals.getData();
    mesh.texcoords = (float2*)meshTexcoords.getData();

    mesh.triangleCount = triangleCount;
    mesh.vertexCount = vertexCount;

    sceneBounds = meshHost->getBounds();
    scenePath = path;
}

Box3f DeviceCuda::getSceneBounds()
{
    return sceneBounds;
}

void DeviceCuda::initCamera(const Props& props)
{
    /*
    std::string type = props.get<std::string>("type");

    if (type == "pinhole")
        impl->camera = makeRef<PinholeCamera>(props);
    else if (type == "thinlens")
        impl->camera = makeRef<ThinLensCamera>(props);
    else
        throw std::invalid_argument("invalid camera type");
    */

    ThinLensCamera cameraHost(props);

    camera.origin = make_float3(cameraHost.origin);
    camera.imageO = make_float3(cameraHost.imageO);
    camera.imageDx = make_float3(cameraHost.imageDx);
    camera.imageDy = make_float3(cameraHost.imageDy);

    camera.basisU = make_float3(cameraHost.basis.U);
    camera.basisV = make_float3(cameraHost.basis.V);

    camera.lensRadius = cameraHost.lensRadius;
    camera.focalDistance = cameraHost.focalDistance;
}

void DeviceCuda::initFrame(const Vec2i& size)
{
    frameBuffer = makeRef<FrameBufferCuda>(size);
}

void DeviceCuda::initToneMapper(const Props& props)
{
}

void DeviceCuda::clearFrame()
{
    frameBuffer->clear();
}

void DeviceCuda::updateFrame(Surface& surface)
{
    const int* in = (const int*)frameBuffer->map();
    char* out = (char*)surface.data;

    #pragma omp parallel for
    for (int y = 0; y < surface.height; ++y)
        memcpy(out + y * (ptrdiff_t)surface.pitch, in + y * (ptrdiff_t)surface.width, surface.width * sizeof(int));

    frameBuffer->unmap();
}

void DeviceCuda::readFrameHdr(Vec3f* dest)
{
}

} // namespace prt
