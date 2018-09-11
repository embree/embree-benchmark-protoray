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
class Diffuse2RendererStream : public Renderer, IntegratorBase
{
private:
    struct StateIo
    {
        RayStream<streamSize> ray;
        RayStreamChannel<float, streamSize> L;
        RayStreamChannel<int, streamSize> pixelId;
    };

    struct State
    {
        RayStreamChannel<int, streamSize> pathId;
        HitStream<streamSize> hit;
        RayStream<streamSize> shadowRay;
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
    int pass;
    int maxDepth;
    bool isStatic;

public:
    Diffuse2RendererStream(const ref<const Scene>& scene, const ref<IntersectorStream<streamSize>>& intersector, const Props& props)
        : Renderer(props)
    {
        this->scene = scene;
        this->intersector = intersector;
        maxDepth = props.get("maxDepth", 6);

        // Initialize the sampler
        int sampleCount = props.get("spp", 0);
        int pixelCount = imageSize.x * imageSize.y;
        sampler.init(getSampleSize(), sampleCount, pixelCount, seed);

        states.alloc(cpuCount);
        for (int i = 0; i < states.getSize(); ++i)
            states[i] = makeRef<State>();

        pass = 0;
        isStatic = props.exists("static");
    }

    void render(const Camera* camera, FrameBuffer* frameBuffer, Props& stats)
    {
        if (frameBuffer->getSize() != imageSize)
            throw std::invalid_argument("Diffuse2RendererStream: framebuffer size mismatch");

        this->camera = camera;
        this->frameBuffer = frameBuffer;

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
        return sampleDimBaseSize + 4 * maxDepth;
    }

    void renderTile(const Vec2i& tileId, int tid)
    {
        State* state = states[tid].get();
        StateIo* stateI = &state->io[0];
        StateIo* stateO = &state->io[1];

        vfloat throughput = 1.f;

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
            stateI->L.setA(i, zero);
        }

        int depth = 0;
        for (;;)
        {
            // Intersect rays
            //RayHint rayHint = (depth == 0) ? rayHintCoherent : rayHintIncoherent;
            RayHint rayHint = rayHintIncoherent; // always use incoherent to avoid frequency drop on SKX
            intersector->intersect(stateI->ray, state->hit, rayCount, state->rayStats, rayHint);

            // Sort
            int missCount = rayStreamSort(stateI->ray, state->pathId.get(), rayCount);

            // No hits
            for (int i = 0; i < missCount; i += simdSize)
            {
                vbool m = (vint(step) + i) < missCount;
                vint pathId = state->pathId.getA(i);

                vfloat L = stateI->L.get(m, pathId);
                L += throughput * ((depth > 0) ? 0.5f : 1.f); // with MIS weight

                vint pixelId = stateI->pixelId.get(m, pathId);
                if (accum)
                    frameBuffer->getColor().add(m, pixelId, Vec3vf(L));
                else
                    frameBuffer->getColor().set(m, pixelId, Vec3vf(L));
            }

            if (missCount == rayCount) break;
            if (depth == maxDepth)
            {
                for (int i = missCount; i < rayCount; i += simdSize)
                {
                    vbool m = (vint(step) + i) < rayCount;
                    vint pathId = state->pathId.get(i);

                    vfloat L = stateI->L.get(m, pathId);

                    vint pixelId = stateI->pixelId.get(m, pathId);
                    if (accum)
                        frameBuffer->getColor().add(m, pixelId, Vec3vf(L));
                    else
                        frameBuffer->getColor().set(m, pixelId, Vec3vf(L));
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

                // Generate a shadow ray
                Vec2vf s = sampler.get2D(state->sampler, sampleDimBaseSize + 4 * depth);
                ray.init(ctx.p, ctx.getFrame() * cosineSampleHemisphere(s), ctx.eps);
                state->shadowRay.setA(o, ray);

                // Generate an extension ray
                s = sampler.get2D(state->sampler, sampleDimBaseSize + 4 * depth + 2);
                ray.init(ctx.p, ctx.getFrame() * cosineSampleHemisphere(s), ctx.eps);
                stateO->ray.setA(o, ray);

                stateO->pixelId.setA(o, pixelId);
            }

            throughput *= 0.8f;

            // Intersect shadow rays
            intersector->occluded(state->shadowRay, rayCount - missCount, state->rayStats);

            // Final pass
            for (int i = missCount, o = 0; i < rayCount; i += simdSize, o += simdSize)
            {
                vbool m = (vint(step) + i) < rayCount;
                vint pathId = state->pathId.get(i);

                vfloat L = stateI->L.get(m, pathId);
                vfloat shadowRayFar = state->shadowRay.far.getA(o);
                vbool mDirMiss = m & (shadowRayFar > 0.0f);
                set(mDirMiss, L, L + (throughput * 0.5f)); // with MIS weight
                stateO->L.setA(o, L);
            }

            rayCount -= missCount;
            swap(stateI, stateO);
            depth++;
        }
    }

public:
    Props queryRay(const Ray& ray)
    {
        return RendererStream::queryRay(scene, intersector, ray);
    }
};

} // namespace prt
