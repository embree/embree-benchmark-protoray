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

#include "sampling/shape_sampler.h"
#include "pinhole_camera.h"

namespace prt {

class ThinLensCamera : public PinholeCamera
{
public:
    float lensRadius;
    float focalDistance;

public:
    ThinLensCamera(const Props& props)
        : PinholeCamera(props)
    {
        lensRadius = props.get("lensRadius", 0.0f);
        focalDistance = props.get("focalDistance", 1.0f);
    }

    FORCEINLINE void getRay(Ray& ray, const CameraSample& s) const
    {
        Vec2f lens = uniformSampleDisk(s.lens) * lensRadius;
        Vec3f begin = origin + basis * Vec3f(lens.x, lens.y, 0.0f);
        Vec3f end = origin + focalDistance * (imageO + s.image.x * imageDx + s.image.y * imageDy);
        ray.init(begin, normalize(end - begin));
    }

    FORCEINLINE void getRay(RaySimd& ray, const CameraSampleSimd& s) const
    {
        Vec2vf lens = uniformSampleDisk(s.lens) * vfloat(lensRadius);
        Vec3vf begin = Vec3vf(origin) + Basis3vf(basis) * Vec3vf(lens.x, lens.y, 0.0f);
        Vec3vf end = Vec3vf(origin) + vfloat(focalDistance) * (Vec3vf(imageO) + s.image.x * Vec3vf(imageDx) + s.image.y * Vec3vf(imageDy));
        ray.init(begin, normalize(end - begin));
    }
};

} // namespace prt
