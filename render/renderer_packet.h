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
    Vec2i imageSize;
    int pass;
    bool isStatic;

public:
    RendererPacket(const ref<const Scene>& scene, const ref<IntersectorPacket>& intersector, const Props& props)
        : integrator(props)
    {
        this->scene = scene;
        this->intersector = intersector;
        imageSize = props.get<Vec2i>("imageSize");

        // Initialize the sampler
        int sampleCount = 64*1024; // FIXME
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

        Vec2i imageSize = frameBuffer->getSize();
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
                    frameBuffer->add(pixelIndex, color);
                else
                    frameBuffer->set(pixelIndex, color);
            }
        }
    }

public:
    Props queryPixel(const Camera* camera, int x, int y)
    {
        Props result;

        CameraSampleSimd cameraSample;
        cameraSample.lens = zero;

        // Generate a ray through the center of the image plane
        // We need this to compute the depth
        RaySimd centerRay;
        cameraSample.image = Vec2f(0.5f);
        camera->getRay(centerRay, cameraSample);

        // Generate a ray through the pixel
        RaySimd ray;
        cameraSample.image = (Vec2f(x, y) + 0.5f) / toFloat(imageSize);
        camera->getRay(ray, cameraSample);

        // Shoot the ray
        HitSimd hit;
        //ShadingContext ctx;
        RayStats stats;
        intersector->intersect(1, ray, hit, stats);
        if (none(ray.isHit())) return result;
        //scene->postIntersect(ray, hit, ctx);

        // Fill the query result
        //result.set("mat", ctx->scene->getMaterialName(ctx));
        //result.set("matId", ctx.matId);
        //result.set("prim", hit.id);
        result.set("depth", toScalar(ray.far * dot(ray.dir, centerRay.dir)));
        //result.set("p", ray.getHitPoint());
        //result.set("Ng", ctx.Ng);
        //result.set("N", ctx.N);
        //result.set("uv", ctx.uv);
        //result.set("U", ctx.U);
        //result.set("V", ctx.V);

        return result;
    }
};

} // namespace prt
