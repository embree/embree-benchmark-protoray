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

#include "core/ray_cuda.cuh"
#include "pinhole_camera_cuda.h"

namespace prt {

CUDA_DEV_FORCEINLINE void getRay(const PinholeCameraCuda& camera, RayCuda& ray, const CameraSampleCuda& s)
{
    float3 dir = normalize(camera.imageO + s.image.x * camera.imageDx + s.image.y * camera.imageDy);
    ray.init(camera.origin, dir);
}

} // namespace prt
