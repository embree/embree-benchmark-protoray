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

#include "sys/logging.h"
#include "sys/array.h"
#include "sys/tasking.h"
#include "sys/sysinfo.h"
#include "sys/timer.h"
#include "sys/logging.h"
#include "image/pixel.h"
#include "image/morton.h"
#include "core/intersector_packet.h"
#include "renderer.h"
#include "integrator.h"

namespace prt {

template <class Integrator, class Sampler, bool accum = true, int tileSizeX = 8, int tileSizeY = 8, int spp = 1>
class RendererPacket : public Renderer
{
private:
    Integrator integrator;
    Sampler sampler;
    ref<IntersectorPacket> intersector;
    ref<const Scene> scene;
    const Camera* camera;
    FrameBuffer* frameBuffer;
    Array<IntegratorState<Sampler>> states;
    int pass;
    bool isStatic;

public:
    RendererPacket(const ref<const Scene>& scene, const ref<IntersectorPacket>& intersector, const Props& props)
        : Renderer(props),
          integrator(props)
    {
        this->scene = scene;
        this->intersector = intersector;

        // Initialize the sampler
        int sampleCount = props.get("spp", 0);
        int pixelCount = imageSize.x * imageSize.y;
        sampler.init(integrator.getSampleSize(), sampleCount, pixelCount);

        states.alloc(cpuCount);

        pass = 0;
        isStatic = props.exists("static");

        Log() << "Tile size: " << tileSizeX << "x" << tileSizeY;
    }

    void render(const Camera* camera, FrameBuffer* frameBuffer, Props& stats)
    {
        if (frameBuffer->getSize() != imageSize)
            throw std::invalid_argument("RendererPacket: framebuffer size mismatch");

        this->camera = camera;
        this->frameBuffer = frameBuffer;

        Vec2i gridSize = imageSize / Vec2i(tileSizeX, tileSizeY);
        Timer timer;
        Tasking::run(gridSize, [this](const Vec2i& tileId, int tid) { renderTile(tileId, tid); });
        double totalTime = timer.query();

        // Stats
        RayStats rayStats;
        for (auto& state : states)
        {
            rayStats += state.rayStats;
            state.rayStats.reset();
        }

        double mray = (double)rayStats.rayCount / 1000000.0 / totalTime;
        stats.set("mray", mray);
        stats.set("ray", rayStats.rayCount);
        stats.set("spp", spp);

#ifdef PROFILE_MODE
        stats.set("nodes", (double)rayStats.nodeCount / rayStats.rayCount);
        stats.set("prims", (double)rayStats.primCount / rayStats.rayCount);
        stats.set("shadeSimdUtil", (double)rayStats.shadeSimdActiveLaneCount / ((double)rayStats.shadeSimdBatchCount * simdSize) * 100.0);
#endif
        if (!isStatic)
            pass += spp;
    }

private:
    void renderTile(const Vec2i& tileId, int tid)
    {
        IntegratorState<Sampler>& state = states[tid];
        Vec2i tileLow = tileId * Vec2i(tileSizeX, tileSizeY);

        for (int sampleId = 0; sampleId < spp; ++sampleId)
        {
            for (int i = 0; i < tileSizeX*tileSizeY; i += simdSize)
            {
                prefetchL1(camera); // workaround for strange slowdown on MIC

                Vec2vi pixel;
                pixel.x = load(mortonTableX + i) + tileLow.x;
                pixel.y = load(mortonTableY + i) + tileLow.y;
                vint pixelIndex = pixel.x + pixel.y * frameBuffer->getSize().x;
                sampler.setSample(one, state.sampler, pass + sampleId, pixelIndex);

                CameraSampleSimd cameraSample;
                Vec2vf pixelSample = sampler.get2D(state.sampler, Integrator::sampleDimPixel);
                cameraSample.image = (toFloat(pixel) + pixelSample) * Vec2vf(frameBuffer->getInvSize());
                cameraSample.lens = sampler.get2D(state.sampler, Integrator::sampleDimLens);

                RaySimd ray;
                camera->getRay(ray, cameraSample);

                Vec3vf color = integrator.getColor(ray, intersector.get(), scene.get(), sampler, state);
                if (accum)
                    frameBuffer->getColor().add(pixelIndex, color);
                else
                    frameBuffer->getColor().set(pixelIndex, color);
            }
        }
    }

public:
    Props queryRay(const Ray& inputRay)
    {
        Props result;

        // Shoot the ray
        RaySimd ray;
        ray.org = inputRay.org;
        ray.dir = inputRay.dir;
        ray.far = inputRay.far;
        HitSimd hit;
        ShadingContextSimd ctx;
        RayStats stats;
        intersector->intersect(1, ray, hit, stats);
        if (none(ray.isHit())) return result;
        scene->postIntersect(1, ray, hit, ctx);
        int primId = *hit.getPrimId();
        int matId = scene->getMaterialId(primId);

        // Fill the query result
        result.set("mat", scene->getMaterialName(matId));
        result.set("matId", matId);
        result.set("prim", primId);
        result.set("dist", toScalar(ray.far));
        result.set("p", toScalar(ray.getHitPoint()));
        result.set("Ng", toScalar(ctx.Ng));
        result.set("N", toScalar(ctx.f.N));
        result.set("uv", toScalar(ctx.uv));
        result.set("U", toScalar(ctx.f.U));
        result.set("V", toScalar(ctx.f.V));
        result.set("eps", toScalar(ctx.eps));

        return result;
    }
};

} // namespace prt
