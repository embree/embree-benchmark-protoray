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

#include "sys/props.h"
#include "math/vec2.h"
#include "math/basis3.h"
#include "camera.h"

namespace prt {

class PinholeCamera : public Camera
{
public:
    Vec3f imageO;
    Vec3f imageDx;
    Vec3f imageDy;

public:
    PinholeCamera(const Props& props)
    {
        Vec3f position = props.get<Vec3f>("position");
        basis = props.get<Basis3f>("basis");
        float fov = props.get<float>("fov");
        float aspectRatio = props.get<float>("aspectRatio");

        float imageHalfHeight = tan(0.5f*fov);
        float imageHalfWidth = imageHalfHeight * aspectRatio;

        origin = position;
        imageO = basis * Vec3f(-imageHalfWidth, imageHalfHeight, -1.0f);
        imageDx = basis * Vec3f(2.0f*imageHalfWidth, 0.0f, 0.0f);
        imageDy = basis * Vec3f(0.0f, -2.0*imageHalfHeight, 0.0f);
    }

    FORCEINLINE void getRay(Ray& ray, const CameraSample& s) const
    {
        Vec3f dir = normalize(imageO + s.image.x * imageDx + s.image.y * imageDy);
        ray.init(origin, dir);
    }

    FORCEINLINE void getRay(RaySimd& ray, const CameraSampleSimd& s) const
    {
        Vec3vf dir = normalize(Vec3vf(imageO) + s.image.x * Vec3vf(imageDx) + s.image.y * Vec3vf(imageDy));
        ray.init(origin, dir);
    }
};

} // namespace prt
