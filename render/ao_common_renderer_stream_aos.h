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

// Shoots AO rays with common origin per stream
template <class ShadingContextT, class Sampler, bool accum, int streamSize, int tileSizeX, int tileSizeY>
class AoCommonRendererStreamAos : public Renderer, IntegratorBase
{
private:
    struct State
    {
        RayHitStreamAos<streamSize> ray;
        RayStreamAos<streamSize> shadowRay;
        RayStreamChannelAos<int, streamSize> pixelId;
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
    AoCommonRendererStreamAos(const ref<const Scene>& scene, const ref<IntersectorStreamAos<streamSize>>& intersector, const Props& props)
        : Renderer(props)
    {
        this->scene = scene;
        this->intersector = intersector;

        sampleCount = props.get("samples", 16);
        if (sampleCount > streamSize)
            throw std::invalid_argument("AoCommonRendererStreamAos: too many samples");

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
            throw std::invalid_argument("AoCommonRendererStreamAos: framebuffer size mismatch");

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

        for (int i = 0; i < rayCount; ++i)
        {
            Ray ray;
            Hit hit;
            state->ray.getRay(i, ray);
            state->ray.getHit(i, hit);

            Vec3f color;

            if (ray.isHit())
            {
                ShadingContextT ctx;
                scene->postIntersect(ray, hit, ctx);

                int pixelId = state->pixelId[i];
                sampler.resetSample(state->sampler, pass, pixelId);

                for (int j = 0; j < sampleCount; ++j)
                {
                    Ray shadowRay;
                    Vec2f s = sampler.get2D(state->sampler, sampleDimBaseSize + 2 * j);
                    shadowRay.init(ctx.p, ctx.getFrame() * cosineSampleHemisphere(s), ctx.eps);
                    state->shadowRay.set(j, shadowRay);
                }

                intersector->occluded(state->shadowRay, sampleCount, state->rayStats);

                float sum = zero;
                for (int j = 0; j < sampleCount; ++j)
                {
                    if (state->shadowRay.isNotOccluded(j))
                        sum += 1.0f;
                }

                color = sum * rcp(float(sampleCount));
            }
            else
            {
                color = zero;
            }

            int pixelId = state->pixelId[i];
            if (accum)
                frameBuffer->getColor().add(pixelId, color);
            else
                frameBuffer->getColor().set(pixelId, color);
        }
    }

public:
    Props queryRay(const Camera* camera, const Ray& ray)
    {
        return RendererStreamAos::queryRay(intersector, ray);
    }
};

} // namespace prt
