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

#include "math/vec2.h"
#include "math/basis3.h"
#include "core/ray.h"
#include "core/ray_simd.h"

namespace prt {

struct CameraSample
{
    Vec2f image;
    Vec2f lens;

    FORCEINLINE CameraSample() {}
    FORCEINLINE CameraSample(const Vec2f& image, const Vec2f& lens) : image(image), lens(lens) {}
};

struct CameraSampleSimd
{
    Vec2vf image;
    Vec2vf lens;

    FORCEINLINE CameraSampleSimd() {}
    FORCEINLINE CameraSampleSimd(const Vec2vf& image, const Vec2vf& lens) : image(image), lens(lens) {}
};

class Camera
{
public:
    Vec3f origin;
    Basis3f basis;

    virtual ~Camera() {}

    virtual void getRay(Ray& ray, const CameraSample& s) const = 0;
    virtual void getRay(RaySimd& ray, const CameraSampleSimd& s) const = 0;
};

} // namespace prt
