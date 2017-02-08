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

#include "intersector.h"
#include "ray_simd.h"

namespace prt {

class IntersectorPacket
{
public:
    virtual ~IntersectorPacket() {}

    virtual void intersect(vbool mask, RaySimd& ray, HitSimd& hit, RayStats& stats) = 0;
    virtual void occluded(vbool mask, RaySimd& ray, RayStats& stats) = 0;
};

} // namespace prt
