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
class PrimaryRendererStream : public Renderer, IntegratorBase
{
private:
    struct State
    {
        RayStream<streamSize> ray;
        RayStreamChannel<int, streamSize> pixelId;
        HitStream<streamSize> hit;
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
    bool isStatic;

public:
    PrimaryRendererStream(const ref<const Scene>& scene, const ref<IntersectorStream<streamSize>>& intersector, const Props& props)
    {
        this->scene = scene;
        this->intersector = intersector;
        imageSize = props.get<Vec2i>("imageSize");

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
            throw std::invalid_argument("PrimaryRendererStream: framebuffer size mismatch");

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
        return sampleDimBaseSize;
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
            /*
            Vec2vi pixel;
            pixel.x = load(mortonTableX + i) + tileLow.x;
            pixel.y = load(mortonTableY + i) + tileLow.y;
            */
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

        for (int i = 0; i < rayCount; i += simdSize)
        {
            RaySimd ray;
            HitSimd hit;
            state->ray.getA(i, ray);
            state->hit.getA(i, hit);

            vbool m = ray.isHit();

            ShadingContextT ctx;
            scene->postIntersect(m, ray, hit, ctx);

            Vec3vf color = (ctx.getN() + vfloat(1.0f)) * vfloat(0.5f);
            color = select(m, color, Vec3vf(0.05f));

            vint pixelId = state->pixelId.getA(i);
            if (accum)
                frameBuffer->add(m, pixelId, color);
            else
                frameBuffer->set(m, pixelId, color);
        }
    }

public:
    Props queryPixel(const Camera* camera, int x, int y)
    {
        return RendererStream::queryPixel(intersector, imageSize, camera, x, y);
    }
};

} // namespace prt
