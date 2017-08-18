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

#ifdef EMBREE_SUPPORT

#include <embree2/rtcore_ray.h>

#else

#include "sys/memory.h"

#define RTC_INVALID_GEOMETRY_ID ((unsigned int)-1)

struct ALIGNED(16) RTCRay
{
    // Ray data
    float org[3];        // origin
    float align0;

    float dir[3];        // direction
    float align1;

    float tnear;         // start of ray segment
    float tfar;          // end of ray segment (set to hit distance)

    float time;          // time (used for motion blur)
    unsigned int mask;   // used to mask out objects

    // Hit data
    float Ng[3];         // unnormalized geometry normal
    float align2;

    float u;             // barycentric u coordinate
    float v;             // barycentric v coordinate

    unsigned int geomID; // geometry ID
    unsigned int primID; // primitive ID
    unsigned int instID; // instance ID
};

#endif
