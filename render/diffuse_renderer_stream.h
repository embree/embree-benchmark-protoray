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
#include "image/pixel.h"
#include "image/morton.h"
#include "sampling/shape_sampler.h"
#include "renderer_stream.h"

//#include "/opt/iaca/include/iacaMarks.h"

namespace prt {

template <class ShadingContextT, class Sampler, bool accum, int streamSize, int tileSizeX, int tileSizeY>
class DiffuseRendererStream : public Renderer, IntegratorBase
{
private:
    struct StateIo
    {
        RayStream<streamSize> ray;
        RayStreamChannel<float, streamSize> throughput;
        RayStreamChannel<int, streamSize> pixelId;
    };

    struct State
    {
        RayStreamChannel<int, streamSize> pathId;
        HitStream<streamSize> hit;
        StateIo io[2];
        typename Sampler::State sampler;
        RayStats rayStats;
    };

    Sampler sampler;
    ref<IntersectorStream<streamSize>> intersector;
    ref<const Scene> scene;
    const Camera* camera;
    FrameBuffer* frameBuffer;
    Array<ref<State>> states;
    Vec2i imageSize;
    int pass;
    int maxDepth;
    bool isStatic;

public:
    DiffuseRendererStream(const ref<const Scene>& scene, const ref<IntersectorStream<streamSize>>& intersector, const Props& props)
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
            throw std::invalid_argument("DiffuseRendererStream: framebuffer size mismatch");

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
        for (int i = 0; i < rayCount; i += simdSize)
        {
            //Vec2vi pixel;
            //pixel.x = load(mortonTableX + i) + tileLow.x;
            //pixel.y = load(mortonTableY + i) + tileLow.y;
            Vec2vi pixel = getMorton8x8Step<tileSizeX>(i) + Vec2vi(tileLow);
            vint pixelId = pixel.x + pixel.y * frameBuffer->getSize().x;
            stateI->pixelId.setA(i, pixelId);

            sampler.setSample(one, state->sampler, pass, pixelId);

            CameraSampleSimd cameraSample;
            Vec2vf pixelSample = sampler.get2D(state->sampler, sampleDimPixel);
            cameraSample.image = (toFloat(pixel) + pixelSample) * Vec2vf(frameBuffer->getInvSize());
            cameraSample.lens = sampler.get2D(state->sampler, sampleDimLens);

            RaySimd ray;
            camera->getRay(ray, cameraSample);

            stateI->ray.setA(i, ray);
            stateI->throughput.setA(i, 1.0f);
        }

        //int matCount = scene->getMaterialCount();
        int matCount = 0; // FIXME!!!!!!!!!

        int depth = 0;
        for (;;)
        {
            // Intersect rays
            intersector->intersect(stateI->ray, state->hit, rayCount, state->rayStats);

            // Sort
            int missCount = rayStreamSort(stateI->ray, state->pathId.get(), rayCount);

            // No hits
            for (int i = 0; i < missCount; i += simdSize)
            {
                vbool m = (vint(step) + i) < missCount;
                vint pathId = state->pathId.get(i);

                vint pixelId = stateI->pixelId.get(m, pathId);
                vfloat throughput = stateI->throughput.get(m, pathId);
                if (accum)
                    frameBuffer->add(m, pixelId, Vec3vf(throughput));
                else
                    frameBuffer->set(m, pixelId, Vec3vf(throughput));
            }

            if (missCount == rayCount) break;
            if (depth == maxDepth)
            {
                for (int i = missCount; i < rayCount; i += simdSize)
                {
                    vbool m = (vint(step) + i) < rayCount;
                    vint pathId = state->pathId.get(i);
                    vint pixelId = stateI->pixelId.get(m, pathId);
                    if (accum)
                        frameBuffer->add(m, pixelId, Vec3vf(zero));
                    else
                        frameBuffer->set(m, pixelId, Vec3vf(zero));
                }
                break;
            }

            // Hits
            for (int i = missCount, o = 0; i < rayCount; i += simdSize, o += simdSize)
            {
                vbool m = (vint(step) + i) < rayCount;
                vint pathId = state->pathId.get(i);

                RaySimd ray;
                HitSimd hit;
                stateI->ray.get(m, pathId, ray);
                state->hit.get(m, pathId, hit);

                ShadingContextT ctx;
                scene->postIntersect(m, ray, hit, ctx);

                vint pixelId = stateI->pixelId.get(m, pathId);
                sampler.resetSample(m, state->sampler, pass, pixelId);
                Vec2vf s = sampler.get2D(state->sampler, sampleDimBaseSize + 2 * depth);
                ray.init(ctx.p, ctx.getBasis() * cosineSampleHemisphere(s), ctx.eps);
                stateO->ray.set(o, ray);

                vfloat throughput = stateI->throughput.get(m, pathId);
                throughput *= 0.8f;
                stateO->throughput.set(o, throughput);
                stateO->pixelId.set(o, pixelId);
            }

            rayCount -= missCount;
            swap(stateI, stateO);
            depth++;
        }
    }

public:
    Props queryPixel(const Camera* camera, int x, int y)
    {
        return RendererStream::queryPixel(intersector, imageSize, camera, x, y);
    }
};

} // namespace prt
