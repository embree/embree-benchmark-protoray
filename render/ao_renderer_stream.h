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

// Shoots one AO ray per pixel per stream
template <class ShadingContextT, class Sampler, bool accum, int streamSize, int tileSizeX, int tileSizeY>
class AoRendererStream : public Renderer, IntegratorBase
{
private:
    struct State
    {
        RayStream<streamSize> ray;
        HitStream<streamSize> hit;
        RayStreamChannel<int, streamSize> pixelId;
        RayStreamChannel<int, streamSize> pathId;
        RayStreamChannel3<float, streamSize> p;
        RayStreamChannel3<float, streamSize> U;
        RayStreamChannel3<float, streamSize> V;
        RayStreamChannel3<float, streamSize> N;
        RayStreamChannel<float, streamSize> eps;
        RayStreamChannel<float, streamSize> throughput;
        typename Sampler::State sampler;
        RayStats rayStats;
    };

    Sampler sampler;
    ref<IntersectorStream<streamSize>> intersector;
    ref<const Scene> scene;
    const Camera* camera;
    FrameBuffer* frameBuffer;
    Array<ref<State>> states;
    int sampleCount;
    int pass;
    bool isStatic;

public:
    AoRendererStream(const ref<const Scene>& scene, const ref<IntersectorStream<streamSize>>& intersector, const Props& props)
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
            throw std::invalid_argument("AoRendererStream: framebuffer size mismatch");

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

        for (int i = 0; i < rayCount; i += simdSize)
        {
            //Vec2vi pixel;
            //pixel.x = load(mortonTableX + i) + tileLow.x;
            //pixel.y = load(mortonTableY + i) + tileLow.y;
            Vec2vi pixel = getMorton8x8Step<tileSizeX>(i) + Vec2vi(tileLow);
            vint pixelId = pixel.x + pixel.y * frameBuffer->getSize().x;
            state->pixelId.setA(i, pixelId);

            sampler.setSample(one, state->sampler, pass, pixelId);

            CameraSampleSimd cameraSample;
            Vec2vf pixelSample = sampler.get2D(state->sampler, sampleDimPixel);
            cameraSample.image = (toFloat(pixel) + pixelSample) * Vec2vf(frameBuffer->getInvSize());
            cameraSample.lens = sampler.get2D(state->sampler, sampleDimLens);

            RaySimd ray;
            camera->getRay(ray, cameraSample);

            state->ray.setA(i, ray);
        }

        intersector->intersect(state->ray, state->hit, rayCount, state->rayStats, rayHintCoherent);

        // Sort
        int missCount = rayStreamSort(state->ray, state->pathId.get(), rayCount);

        // No hits
        for (int i = 0; i < missCount; i += simdSize)
        {
            vbool m = (vint(step) + i) < missCount;
            vint pathId = state->pathId.getA(i);

            vint pixelId = state->pixelId.get(m, pathId);
            const vfloat color = zero;

            if (accum)
                frameBuffer->getColor().add(m, pixelId, Vec3vf(color));
            else
                frameBuffer->getColor().set(m, pixelId, Vec3vf(color));
        }

        // Hits
        for (int i = missCount, o = 0; i < rayCount; i += simdSize, o += simdSize)
        {
            vbool m = (vint(step) + i) < rayCount;
            vint pathId = state->pathId.get(i);

            RaySimd ray;
            HitSimd hit;
            state->ray.get(m, pathId, ray);
            state->hit.get(m, pathId, hit);

            ShadingContextT ctx;
            scene->postIntersect(m, ray, hit, ctx);
            Basis3vf frame = ctx.getFrame();

            state->p.setA(o, ctx.p);
            state->U.setA(o, frame.U);
            state->V.setA(o, frame.V);
            state->N.setA(o, frame.N);
            state->eps.setA(o, ctx.eps);
            state->throughput.setA(o, zero);
        }

        for (int k = 0; k < sampleCount; ++k)
        {
            for (int i = missCount, o = 0; i < rayCount; i += simdSize, o += simdSize)
            {
                vbool m = (vint(step) + i) < rayCount;
                vint pathId = state->pathId.get(i);

                RaySimd ray;
                Vec3vf p = state->p.getA(o);
                Basis3vf frame(state->U.getA(o), state->V.getA(o), state->N.getA(o));
                vfloat eps = state->eps.getA(o);

                vint pixelId = state->pixelId.get(m, pathId);
                sampler.resetSample(m, state->sampler, pass, pixelId);
                Vec2vf s = sampler.get2D(state->sampler, sampleDimBaseSize + 2 * k);
                ray.init(p, frame * cosineSampleHemisphere(s), eps);
                state->ray.setA(o, ray);
            }

            intersector->occluded(state->ray, rayCount - missCount, state->rayStats);

            for (int i = missCount, o = 0; i < rayCount; i += simdSize, o += simdSize)
            {
                vbool mHit = state->ray.far.getA(o) == 0.f;
                vfloat throughput = state->throughput.getA(o) + select(mHit, vfloat(0.f), vfloat(1.f));
                state->throughput.setA(o, throughput);
            }
        }

        // Final pass
        for (int i = missCount, o = 0; i < rayCount; i += simdSize, o += simdSize)
        {
            vbool m = (vint(step) + i) < rayCount;
            vint pathId = state->pathId.get(i);
            vint pixelId = state->pixelId.get(m, pathId);
            vfloat color = state->throughput.getA(o) * rcp(float(sampleCount));

            if (accum)
                frameBuffer->getColor().add(m, pixelId, Vec3vf(color));
            else
                frameBuffer->getColor().set(m, pixelId, Vec3vf(color));
        }
    }

public:
    Props queryRay(const Ray& ray)
    {
        return RendererStream::queryRay(scene, intersector, ray);
    }
};

} // namespace prt
