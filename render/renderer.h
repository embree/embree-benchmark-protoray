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

#include <ctime>
#include "sys/props.h"
#include "sys/logging.h"
#include "math/random.h"
#include "camera/camera.h"
#include "image/frame_buffer.h"
#include "scene.h"

namespace prt {

class Renderer
{
protected:
    Vec2i imageSize;
    int seed;

public:
    Renderer(const Props& props)
    {
        imageSize = props.get<Vec2i>("imageSize");

        if (props.exists("randomSeed"))
        {
            Log() << "Generating random seed";
            seed = generateRandomSeed();
        }
        else
        {
            seed = 0;
        }
    }

    virtual ~Renderer() {}

    virtual void render(const Camera* camera, FrameBuffer* frameBuffer, Props& stats) = 0;

    virtual Props queryRay(const Ray& ray) { return Props(); }

    virtual Props queryPixel(const Camera* camera, int x, int y)
    {
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
        Props result = queryRay(ray);
        if (result.isEmpty()) return result;

        // Fill the query result
        float dist = result.get<float>("dist");
        result.set("depth", dist * dot(ray.dir, centerRay.dir));
        return result;
    }
};

} // namespace prt
