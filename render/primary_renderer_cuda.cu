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

#include <stdexcept>
#include "core/intersector_cuda.cuh"
#include "geometry/triangle_mesh_cuda.cuh"
#include "sampling/random_sampler.cuh"
#include "camera/pinhole_camera_cuda.cuh"
#include "camera/thin_lens_camera_cuda.cuh"
#include "primary_renderer_cuda.h"

namespace prt {

template <class CameraCuda>
static CUDA_DEV_KERNEL void generateRaysKernel(CameraCuda camera,
                                               AccumBufferCuda accumBuffer,
                                               int pass,
                                               RayCuda* rays,
                                               int* pixelIds)
{
    // Generate rays in Morton order
    int tx = threadIdx.x;
    int mx = (tx & 1) | ((tx & 4) >> 1) | ((tx & 16) >> 2);
    int my = ((tx & 2) >> 1) | ((tx & 8) >> 2);
    int x = blockIdx.x * 8 + mx;
    int y = blockIdx.y * 16 + threadIdx.y * 4 + my;

    int pixelId = x + y * accumBuffer.size.x;

    RandomSampler sampler;
    sampler.init(pass, pixelId);

    CameraSampleCuda cameraSample;
    float2 pixelSample = sampler.get2D();
    cameraSample.image = (make_float2(x, y) + pixelSample) / make_float2(accumBuffer.size.x, accumBuffer.size.y);
    cameraSample.lens = sampler.get2D();

    RayCuda ray;
    getRay(camera, ray, cameraSample);

    int i = (blockDim.x * blockDim.y) * (blockIdx.y * gridDim.x + blockIdx.x) + (threadIdx.y * blockDim.x + threadIdx.x);
    rays[i] = ray;
    pixelIds[i] = pixelId;
}

template <class ShadingContextT, bool isAccum>
static CUDA_DEV_KERNEL void shadeRaysKernel(TriangleMeshCuda mesh,
                                            AccumBufferCuda accumBuffer,
                                            const RayCuda* rays,
                                            const HitCuda* hits,
                                            const int* pixelIds,
                                            int count)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count)
        return;

    HitCuda hit = hits[i];

    float3 color;
    if (hit.isHit())
    {
        RayCuda ray = rays[i];
        ShadingContextT ctx;
        postIntersect(mesh, ray, hit, ctx);
        color = (ctx.getN() + make_float3(1.0f, 1.0f, 1.0f)) * 0.5f;
    }
    else
    {
        color = make_float3(0.05f, 0.05f, 0.05f);
    }

    int pixelId = pixelIds[i];
    if (isAccum)
    {
        float4 accum = accumBuffer.data[pixelId];
        accumBuffer.data[pixelId] = make_float4(color.x+accum.x, color.y+accum.y, color.z+accum.z, 1.0f+accum.w);
    }
    else
    {
        accumBuffer.data[pixelId] = make_float4(color.x, color.y, color.z, 1.0f);
    }
}

PrimaryRendererCuda::PrimaryRendererCuda(const TriangleMeshCuda& mesh, IntersectorStreamCuda* intersector, int imageSize, bool isFast)
    : mesh(mesh),
      intersector(intersector),
      pixelCount(imageSize),
      pass(0),
      isFast(isFast)
{
    cudaMalloc(&rays, imageSize * sizeof(RayCuda));
    cudaMalloc(&hits, imageSize * sizeof(HitCuda));
    cudaMalloc(&pixelIds, imageSize * sizeof(int));

    cudaFuncSetAttribute(generateRaysKernel<PinholeCameraCuda>,            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    cudaFuncSetAttribute(generateRaysKernel<ThinLensCameraCuda>,           cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    cudaFuncSetAttribute(shadeRaysKernel<ShadingContextCuda, true>,        cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    cudaFuncSetAttribute(shadeRaysKernel<SimpleShadingContextCuda, false>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
}

PrimaryRendererCuda::~PrimaryRendererCuda()
{
    cudaFree(rays);
    cudaFree(hits);
    cudaFree(pixelIds);

    delete intersector;
}

int PrimaryRendererCuda::render(const ThinLensCameraCuda& camera, const AccumBufferCuda& accumBuffer)
{   
    if (accumBuffer.size.x % 8 != 0 || accumBuffer.size.y % 16 != 0)
        throw std::logic_error("image size is not divisible by the tile size");

    int count = pixelCount;

    // Generate camera rays
    dim3 genBlockSize(32, 4);
    dim3 genGridSize(accumBuffer.size.x / 8, accumBuffer.size.y / 16);
    if (camera.lensRadius == 0.0f)
        generateRaysKernel<<<genGridSize, genBlockSize>>>((const PinholeCameraCuda&)camera, accumBuffer, pass, rays, pixelIds);
    else
        generateRaysKernel<<<genGridSize, genBlockSize>>>(camera, accumBuffer, pass, rays, pixelIds);

    // Intersect the rays
    intersector->intersect(rays, hits, count);

    // Shade the rays
    dim3 blockSize(128);
    dim3 gridSize((count + blockSize.x - 1) / blockSize.x);

    if (isFast)
        shadeRaysKernel<SimpleShadingContextCuda, false><<<gridSize, blockSize>>>(mesh, accumBuffer, rays, hits, pixelIds, count);
    else
        shadeRaysKernel<ShadingContextCuda, true><<<gridSize, blockSize>>>(mesh, accumBuffer, rays, hits, pixelIds, count);

    pass++;
    return count;
}

} // namespace prt
