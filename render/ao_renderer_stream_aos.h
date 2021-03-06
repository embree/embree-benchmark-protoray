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
#include "renderer_stream_aos.h"

//#include "/opt/iaca/include/iacaMarks.h"

namespace prt {

// Shoots one AO ray per pixel per stream
template <class ShadingContextT, class Sampler, bool accum, int streamSize, int tileSizeX, int tileSizeY>
class AoRendererStreamAos : public Renderer, IntegratorBase
{
private:
    struct AoShadingContext
    {
        Vec3f p;
        Basis3f f;
        float eps;
    };

    struct State
    {
        RayHitStreamAos<streamSize> ray;
        RayStreamAos<streamSize> shadowRay;
        RayStreamChannelAos<int, streamSize> pixelId;
        RayStreamChannelAos<int, streamSize> pathId;
        RayStreamChannelAos<AoShadingContext, streamSize> ctx;
        RayStreamChannelAos<float, streamSize> throughput;
        typename Sampler::State sampler;
        RayStats rayStats;
    };

    Sampler sampler;
    ref<IntersectorStreamAos<streamSize>> intersector;
    ref<const Scene> scene;
    const Camera* camera;
    FrameBuffer* frameBuffer;
    Array<ref<State>> states;
    int sampleCount;
    int pass;
    bool isStatic;

public:
    AoRendererStreamAos(const ref<const Scene>& scene, const ref<IntersectorStreamAos<streamSize>>& intersector, const Props& props)
        : Renderer(props)
    {
        this->scene = scene;
        this->intersector = intersector;

        sampleCount = props.get("samples", 16);

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
            throw std::invalid_argument("AoRendererStreamAos: framebuffer size mismatch");

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
        return sampleDimBaseSize + 2 * sampleCount;
    }

    void renderTile(const Vec2i& tileId, int tid)
    {
        State* state = states[tid].get();

        Vec2i tileLow = tileId * Vec2i(tileSizeX, tileSizeY);
        //int rayCount = tileSizeX*tileSizeY;
        int croppedTileSizeY = min(imageSize.y - tileLow.y, tileSizeY);
        int rayCount = tileSizeX*croppedTileSizeY;

        for (int i = 0; i < rayCount; ++i)
        {
            //Vec2vi pixel;
            //pixel.x = load(mortonTableX + i) + tileLow.x;
            //pixel.y = load(mortonTableY + i) + tileLow.y;
            Vec2i pixel = getMorton8x8<tileSizeX>(i) + Vec2i(tileLow);
            int pixelId = pixel.x + pixel.y * frameBuffer->getSize().x;
            state->pixelId[i] = pixelId;

            sampler.setSample(state->sampler, pass, pixelId);

            CameraSample cameraSample;
            Vec2f pixelSample = sampler.get2D(state->sampler, sampleDimPixel);
            cameraSample.image = (toFloat(pixel) + pixelSample) * Vec2f(frameBuffer->getInvSize());
            cameraSample.lens = sampler.get2D(state->sampler, sampleDimLens);

            Ray ray;
            camera->getRay(ray, cameraSample);

            state->ray.set(i, ray);
        }

        intersector->intersect(state->ray, rayCount, state->rayStats, rayHintCoherent);

        // Sort
        int missCount = rayStreamSort(state->ray, state->pathId.get(), rayCount);

        // No hits
        for (int i = 0; i < missCount; ++i)
        {
            int pathId = state->pathId[i];

            int pixelId = state->pixelId[pathId];
            const float color = zero;
            if (accum)
                frameBuffer->getColor().add(pixelId, Vec3f(color));
            else
                frameBuffer->getColor().set(pixelId, Vec3f(color));
        }

        // Hits
        for (int i = missCount, o = 0; i < rayCount; ++i, ++o)
        {
            int pathId = state->pathId[i];

            Ray ray;
            Hit hit;
            state->ray.getRay(pathId, ray);
            state->ray.getHit(pathId, hit);

            ShadingContextT ctx;
            scene->postIntersect(ray, hit, ctx);

            state->ctx[o].p = ctx.p;
            state->ctx[o].f = ctx.getFrame();
            state->ctx[o].eps = ctx.eps;
            state->throughput[o] = zero;
        }

        for (int k = 0; k < sampleCount; ++k)
        {
            for (int i = missCount, o = 0; i < rayCount; ++i, ++o)
            {
                int pathId = state->pathId[i];

                Ray ray;
                Vec3f p = state->ctx[o].p;
                Basis3f frame = state->ctx[o].f;
                float eps = state->ctx[o].eps;

                int pixelId = state->pixelId[pathId];
                sampler.resetSample(state->sampler, pass, pixelId);
                Vec2f s = sampler.get2D(state->sampler, sampleDimBaseSize + 2 * k);
                ray.init(p, frame * cosineSampleHemisphere(s), eps);
                state->shadowRay.set(o, ray);
            }

            intersector->occluded(state->shadowRay, rayCount - missCount, state->rayStats);

            for (int i = missCount, o = 0; i < rayCount; ++i, ++o)
            {
                if (state->shadowRay.isNotOccluded(o))
                    state->throughput[o] += 1.f;
            }
        }

        // Final pass
        for (int i = missCount, o = 0; i < rayCount; ++i, ++o)
        {
            int pathId = state->pathId[i];
            int pixelId = state->pixelId[pathId];
            float color = state->throughput[o] * rcp(float(sampleCount));

            if (accum)
                frameBuffer->getColor().add(pixelId, Vec3f(color));
            else
                frameBuffer->getColor().set(pixelId, Vec3f(color));
        }
    }

public:
    Props queryRay(const Ray& ray)
    {
        return RendererStreamAos::queryRay(scene, intersector, ray);
    }
};

} // namespace prt
