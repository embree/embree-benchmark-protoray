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
#include "diffuse2_renderer_cuda.h"

namespace prt {

template <class CameraCuda>
static CUDA_DEV_KERNEL void generateRaysKernel(CameraCuda camera,
                                               AccumBufferCuda accumBuffer,
                                               int pass,
                                               RayCuda* rays,
                                               int* pixelIds,
                                               unsigned int* samplerStates,
                                               float* L)
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
    samplerStates[i] = sampler.getState();
    L[i] = 0.f;
}

template <class ShadingContextT, bool isAccum>
static CUDA_DEV_KERNEL void shadeRaysKernel(TriangleMeshCuda mesh,
                                            AccumBufferCuda accumBuffer,
                                            const RayCuda* rays, RayCuda* rays_o,
                                            const HitCuda* hits,
                                            RayCuda* shadowRays_o,
                                            const AnyHitCuda* shadowHits,
                                            const int* pixelIds, int* pixelIds_o,
                                            const unsigned int* samplerStates, unsigned int* samplerStates_o,
                                            const float* L, float* L_o,
                                            float throughput,
                                            int* queueSize,
                                            int count,
                                            bool final)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count)
        return;

    float color = L[i];
    if (shadowHits && !shadowHits[i].isHit())
        color += throughput * 0.5f; // with MIS weight

    HitCuda hit = hits[i];
    int pixelId = pixelIds[i];

    bool isHit = hit.isHit();

    if (isHit)
    {
        if (!final)
        {
            int o = atomicIncAgg(queueSize);

            RayCuda ray = rays[i];
            ShadingContextT ctx;
            postIntersect(mesh, ray, hit, ctx);

            RandomSampler sampler;
            sampler.init(samplerStates[i]);

            // Generate shadow ray
            float2 s = sampler.get2D();
            ray.init(ctx.p, ctx.getFrame() * cosineSampleHemisphere(s), ctx.eps);
            shadowRays_o[o] = ray;

            // Generate extension ray
            s = sampler.get2D();
            ray.init(ctx.p, ctx.getFrame() * cosineSampleHemisphere(s), ctx.eps);
            rays_o[o] = ray;

            pixelIds_o[o] = pixelId;
            samplerStates_o[o] = sampler.getState();
            L_o[o] = color;
        }
    }
    else
    {
        color += throughput * (shadowHits ? 0.5f : 1.f); // with MIS weight
    }

    if (!isHit || final)
    {
        if (isAccum)
        {
            float4 accum = accumBuffer.data[pixelId];
            accumBuffer.data[pixelId] = make_float4(color+accum.x, color+accum.y, color+accum.z, 1.0f+accum.w);
        }
        else
        {
            accumBuffer.data[pixelId] = make_float4(color, color, color, 1.0f);
        }
    }
}

Diffuse2RendererCuda::Diffuse2RendererCuda(const TriangleMeshCuda& mesh, IntersectorStreamCuda* intersector, int imageSize, int maxDepth, bool isFast)
    : mesh(mesh),
      intersector(intersector),
      pixelCount(imageSize),
      pass(0),
      maxDepth(maxDepth),
      isFast(isFast)
{
    cudaMalloc(&hits, imageSize * sizeof(HitCuda));
    cudaMalloc(&shadowRays, imageSize * sizeof(RayCuda));
    cudaMalloc(&shadowHits, imageSize * sizeof(AnyHitCuda));
    for (int i = 0; i < 2; ++i)
    {
        cudaMalloc(&rays[i], imageSize * sizeof(RayCuda));
        cudaMalloc(&pixelIds[i], imageSize * sizeof(int));
        cudaMalloc(&samplerStates[i], imageSize * sizeof(int));
        cudaMalloc(&L[i], imageSize * sizeof(float));
    }
    cudaMalloc(&queueSize, sizeof(int));

    cudaFuncSetAttribute(generateRaysKernel<PinholeCameraCuda>,            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    cudaFuncSetAttribute(generateRaysKernel<ThinLensCameraCuda>,           cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    cudaFuncSetAttribute(shadeRaysKernel<ShadingContextCuda, true>,        cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    cudaFuncSetAttribute(shadeRaysKernel<SimpleShadingContextCuda, false>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
}

Diffuse2RendererCuda::~Diffuse2RendererCuda()
{
    cudaFree(hits);
    cudaFree(shadowRays);
    cudaFree(shadowHits);
    for (int i = 0; i < 2; ++i)
    {
        cudaFree(rays[i]);
        cudaFree(pixelIds[i]);
        cudaFree(samplerStates[i]);
        cudaFree(L[i]);
    }
    cudaFree(queueSize);

    delete intersector;
}

int Diffuse2RendererCuda::render(const ThinLensCameraCuda& camera, const AccumBufferCuda& accumBuffer)
{
    if (accumBuffer.size.x % 8 != 0 || accumBuffer.size.y % 16 != 0)
        throw std::logic_error("image size is not divisible by the tile size");

    int totalRayCount = 0;
    int count = pixelCount;
    int buf = 0;

    // Generate camera rays
    dim3 genBlockSize(32, 4);
    dim3 genGridSize(accumBuffer.size.x / 8, accumBuffer.size.y / 16);
    if (camera.lensRadius == 0.0f)
        generateRaysKernel<<<genGridSize, genBlockSize>>>((const PinholeCameraCuda&)camera, accumBuffer, pass, rays[buf], pixelIds[buf], samplerStates[buf], L[buf]);
    else
        generateRaysKernel<<<genGridSize, genBlockSize>>>(camera, accumBuffer, pass, rays[buf], pixelIds[buf], samplerStates[buf], L[buf]);

    dim3 blockSize(256);
    dim3 gridSize((count + blockSize.x - 1) / blockSize.x);

    float throughput = 1.f;

    int depth = 0;
    while (count > 0)
    {
        // Intersect the rays
        intersector->intersect(rays[buf], hits, count);
        totalRayCount += count;
        if (depth > 0)
        {
            intersector->occluded(shadowRays, shadowHits, count);
            totalRayCount += count;
        }

        // Shade the rays
        int emptyQueueSize = 0;
        cudaMemcpy(queueSize, &emptyQueueSize, sizeof(int), cudaMemcpyHostToDevice);

        int buf2 = 1-buf;
        bool final = depth == maxDepth;

        if (isFast)
            shadeRaysKernel<SimpleShadingContextCuda, false><<<gridSize, blockSize>>>(mesh, accumBuffer, rays[buf], rays[buf2], hits, shadowRays, (depth > 0) ? shadowHits : nullptr, pixelIds[buf], pixelIds[buf2], samplerStates[buf], samplerStates[buf2], L[buf], L[buf2], throughput, queueSize, count, final);
        else
            shadeRaysKernel<ShadingContextCuda, true><<<gridSize, blockSize>>>(mesh, accumBuffer, rays[buf], rays[buf2], hits, shadowRays, (depth > 0) ? shadowHits : nullptr, pixelIds[buf], pixelIds[buf2], samplerStates[buf], samplerStates[buf2], L[buf], L[buf2], throughput, queueSize, count, final);

        if (final)
            break;

        throughput *= 0.8f;

        cudaMemcpy(&count, queueSize, sizeof(int), cudaMemcpyDeviceToHost);
        buf = buf2;
        depth++;
    }

    pass++;
    return totalRayCount;
}

} // namespace prt
