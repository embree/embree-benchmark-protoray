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

#include "sys/logging.h"
#include "sys/array.h"
#include "sys/tasking.h"
#include "sys/sysinfo.h"
#include "sys/timer.h"
#include "image/pixel.h"
#include "image/morton.h"
#include "core/intersector_single.h"
#include "renderer.h"
#include "integrator.h"

namespace prt {

template <class Integrator, class Sampler, bool accum = true, int tileSizeX = 8, int tileSizeY = 8, int spp = 1>
class RendererSingle : public Renderer
{
private:
    Integrator integrator;
    Sampler sampler;
    ref<IntersectorSingle> intersector;
    ref<const Scene> scene;
    const Camera* camera;
    FrameBuffer* frameBuffer;
    Array<IntegratorState<Sampler>> states;
    Vec2i imageSize;
    int pass;
    bool isStatic;

public:
    RendererSingle(const ref<const Scene>& scene, const ref<IntersectorSingle>& intersector, const Props& props)
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
            throw std::invalid_argument("RendererSingle: framebuffer size mismatch");

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
            for (int i = 0; i < tileSizeX*tileSizeY; ++i)
            {
                prefetchL1(camera); // workaround for strange slowdown on MIC

                Vec2i pixel = Vec2i(mortonTableX[i], mortonTableY[i]) + tileLow;
                int pixelIndex = pixel.x + pixel.y * frameBuffer->getSize().x;
                sampler.setSample(state.sampler, pass + sampleId, pixelIndex);

                CameraSample cameraSample;
                Vec2f pixelSample = sampler.get2D(state.sampler, Integrator::sampleDimPixel);
                cameraSample.image = (toFloat(pixel) + pixelSample) * frameBuffer->getInvSize();
                cameraSample.lens = sampler.get2D(state.sampler, Integrator::sampleDimLens);

                Ray ray;
                camera->getRay(ray, cameraSample);

                Vec3f color = integrator.getColor(ray, intersector.get(), scene.get(), sampler, state);
                //if (!all(isfinite(color))) LogWarn() << "Infinite radiance: " << color;

                if (accum)
                    frameBuffer->add(pixel, color);
                else
                    frameBuffer->set(pixel, color);
            }
        }
    }

public:
    Props queryPixel(const Camera* camera, int x, int y)
    {
        Props result;

        CameraSample cameraSample;
        cameraSample.lens = zero;

        // Generate a ray through the center of the image plane
        // We need this to compute the depth
        Ray centerRay;
        cameraSample.image = 0.5f;
        camera->getRay(centerRay, cameraSample);

        // Generate a ray through the pixel
        Ray ray;
        cameraSample.image = (Vec2f(x, y) + 0.5f) / toFloat(imageSize);
        camera->getRay(ray, cameraSample);

        // Shoot the ray
        Hit hit;
        ShadingContext ctx;
        RayStats stats;
        intersector->intersect(ray, hit, stats);
        if (!ray.isHit()) return result;
        scene->postIntersect(ray, hit, ctx);
        int matId = scene->getMaterialId(hit.primId);

        // Fill the query result
        result.set("mat", scene->getMaterialName(matId));
        result.set("matId", matId);
        result.set("prim", hit.primId);
        result.set("depth", ray.far * dot(ray.dir, centerRay.dir));
        result.set("p", ray.getHitPoint());
        result.set("Ng", ctx.Ng);
        result.set("N", ctx.N);
        result.set("uv", ctx.uv);

        return result;
    }
};

} // namespace prt
