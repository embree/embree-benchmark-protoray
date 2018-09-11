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
#include "sampling/shape_sampler.cuh"
#include "thin_lens_camera_cuda.h"

namespace prt {

CUDA_DEV_FORCEINLINE void getRay(const ThinLensCameraCuda& camera, RayCuda& ray, const CameraSampleCuda& s)
{
    float2 lens = uniformSampleDisk(s.lens) * camera.lensRadius;
    float3 begin = camera.origin + lens.x * camera.basisU + lens.y * camera.basisV;
    float3 end = camera.origin + camera.focalDistance * (camera.imageO + s.image.x * camera.imageDx + s.image.y * camera.imageDy);
    ray.init(begin, normalize(end - begin));
}

} // namespace prt
