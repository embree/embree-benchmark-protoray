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
#include "core/ray_stream.h"
#include "image/pixel.h"
#include "image/morton.h"
#include "sampling/shape_sampler.h"
#include "renderer_stream_aos.h"

//#include "/opt/iaca/include/iacaMarks.h"

namespace prt {

template <class ShadingContextT, class Sampler, bool accum, int streamSize, int tileSizeX, int tileSizeY>
class DiffuseRendererStreamAos : public Renderer, IntegratorBase
{
private:
    struct StateIo
    {
        RayHitStreamAos<streamSize> ray;
        RayStreamChannel<float, streamSize> throughput;
        RayStreamChannel<int, streamSize> pixelId;
    };

    struct State
    {
        RayStreamChannel<int, streamSize> pathId;
        StateIo io[2];
        typename Sampler::State sampler;
        RayStats rayStats;
    };

    Sampler sampler;
    ref<IntersectorStreamAos<streamSize>> intersector;
    ref<const Scene> scene;
    const Camera* camera;
    FrameBuffer* frameBuffer;
    Array<ref<State>> states;
    Vec2i imageSize;
    int pass;
    int maxDepth;
    bool isStatic;

public:
    DiffuseRendererStreamAos(const ref<const Scene>& scene, const ref<IntersectorStreamAos<streamSize>>& intersector, const Props& props)
    {
        this->scene = scene;
        this->intersector = intersector;
        imageSize = props.get<Vec2i>("imageSize");
        maxDepth = props.get("maxDepth", 6);

        // Initialize the sampler
        int sampleCount = 64*1024; // FIXME
        int pixelCount = imageSize.x * imageSize.y;
        sampler.init(getSampleSize(), sampleCount, pixelCount);

        states.alloc(cpuCount);
        for (int i = 0; i < states.getSize(); ++i)
            states[i] = makeRef<State>();

        pass = 0;
        isStatic = props.exists("static");
    }

    void render(const Camera* camera, FrameBuffer* frameBuffer, Props& stats)
    {
        if (frameBuffer->getSize() != imageSize)
            throw std::invalid_argument("DiffuseRendererStreamAos: framebuffer size mismatch");

        this->camera = camera;
        this->frameBuffer = frameBuffer;

        Vec2i imageSize = frameBuffer->getSize();
        Vec2i tileSize = Vec2i(tileSizeX, tileSizeY);
        Vec2i gridSize = (imageSize + tileSize - 1) / tileSize;
        Timer timer;
        Tasking::run(gridSize, [this](const Vec2i& tileId, int tid) { renderTile(tileId, tid); });
        double totalTime = timer.query();

        // Stats
        RayStats rayStats;
        for (int i = 0; i < states.getSize(); ++i)
        {
            rayStats += states[i]->rayStats;
            states[i]->rayStats.reset();
        }

        double mray = (double)rayStats.rayCount / 1000000.0 / totalTime;
        stats.set("mray", mray);
        stats.set("ray", rayStats.rayCount);

#ifdef PROFILE_MODE
        stats.set("nodes", (double)rayStats.nodeCount / rayStats.rayCount);
        stats.set("prims", (double)rayStats.primCount / rayStats.rayCount);
#endif
        if (!isStatic)
            ++pass;
    }

private:
    int getSampleSize() const
    {
        return sampleDimBaseSize + 2 * maxDepth;
    }

    void renderTile(const Vec2i& tileId, int tid)
    {
        State* state = states[tid].get();
        StateIo* stateI = &state->io[0];
        StateIo* stateO = &state->io[1];

        Vec2i tileLow = tileId * Vec2i(tileSizeX, tileSizeY);
        int croppedTileSizeY = min(imageSize.y - tileLow.y, tileSizeY);

        int rayCount = tileSizeX*croppedTileSizeY;
        for (int i = 0; i < rayCount; ++i)
        {
            //Vec2vi pixel;
            //pixel.x = load(mortonTableX + i) + tileLow.x;
            //pixel.y = load(mortonTableY + i) + tileLow.y;
            Vec2i pixel = getMorton8x8<tileSizeX>(i) + Vec2i(tileLow);
            int pixelId = pixel.x + pixel.y * frameBuffer->getSize().x;
            stateI->pixelId[i] = pixelId;

            sampler.setSample(state->sampler, pass, pixelId);

            CameraSample cameraSample;
            Vec2f pixelSample = sampler.get2D(state->sampler, sampleDimPixel);
            cameraSample.image = (toFloat(pixel) + pixelSample) * Vec2f(frameBuffer->getInvSize());
            cameraSample.lens = sampler.get2D(state->sampler, sampleDimLens);

            Ray ray;
            camera->getRay(ray, cameraSample);

            stateI->ray.set(i, ray);
            stateI->throughput[i] = 1.0f;
        }

        int depth = 0;
        for (;;)
        {
            // Intersect rays
            intersector->intersect(stateI->ray, rayCount, state->rayStats);

            // Sort
            int missCount = rayStreamSort(stateI->ray, state->pathId.get(), rayCount);

            // No hits
            for (int i = 0; i < missCount; ++i)
            {
                int pathId = state->pathId[i];

                int pixelId = stateI->pixelId[pathId];
                float throughput = stateI->throughput[pathId];
                if (accum)
                    frameBuffer->add(pixelId, Vec3f(throughput));
                else
                    frameBuffer->set(pixelId, Vec3f(throughput));
            }

            if (missCount == rayCount) break;
            if (depth == maxDepth)
            {
                for (int i = missCount; i < rayCount; ++i)
                {
                    int pathId = state->pathId[i];
                    int pixelId = stateI->pixelId[pathId];
                    if (accum)
                        frameBuffer->add(pixelId, Vec3f(zero));
                    else
                        frameBuffer->set(pixelId, Vec3f(zero));
                }
                break;
            }

            // Hits
            for (int i = missCount, o = 0; i < rayCount; ++i, ++o)
            {
                int pathId = state->pathId[i];

                Ray ray;
                Hit hit;
                stateI->ray.getRay(pathId, ray);
                stateI->ray.getHit(pathId, hit);

                ShadingContextT ctx;
                scene->postIntersect(ray, hit, ctx);

                int pixelId = stateI->pixelId[pathId];
                sampler.resetSample(state->sampler, pass, pixelId);
                Vec2f s = sampler.get2D(state->sampler, sampleDimBaseSize + 2 * depth);
                ray.init(ctx.p, ctx.getBasis() * cosineSampleHemisphere(s), ctx.eps);
                stateO->ray.set(o, ray);

                float throughput = stateI->throughput[pathId];
                throughput *= 0.8f;
                stateO->throughput[o] = throughput;
                stateO->pixelId[o] = pixelId;
            }

            rayCount -= missCount;
            swap(stateI, stateO);
            depth++;
        }
    }

public:
    Props queryPixel(const Camera* camera, int x, int y)
    {
        return RendererStreamAos::queryPixel(intersector, imageSize, camera, x, y);
    }
};

} // namespace prt
