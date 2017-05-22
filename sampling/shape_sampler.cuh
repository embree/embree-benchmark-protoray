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

#include "math/math.cuh"

namespace prt {

CUDA_DEV_FORCEINLINE float3 cosineSampleHemisphere(const float2& s)
{
    float cosTheta = sqrt(s.x);
    float sinTheta = sqrt(1.0f-s.x);

    float phi = 2.0f*float(pi) * s.y;

    //pdf = cosTheta * (1.0f/float(pi));

    float sinPhi, cosPhi;
    sincos(phi, &sinPhi, &cosPhi);

    float x = cosPhi * sinTheta;
    float y = sinPhi * sinTheta;
    float z = cosTheta;
    return make_float3(x,y,z);
}

CUDA_DEV_FORCEINLINE float2 uniformSampleDisk(const float2& s)
{
    float r = sqrt(s.x);
    float theta = 2.0f*float(pi) * s.y;

    float sinTheta, cosTheta;
    sincos(theta, &sinTheta, &cosTheta);

    return make_float2(r*cosTheta, r*sinTheta);
}

} // namespace prt
