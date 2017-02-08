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

#include "sys/common.h"

namespace prt {

struct RayStats
{
    int64_t rayCount;
    int64_t nodeCount;
    int64_t primCount;

    int64_t shadeSimdBatchCount;
    int64_t shadeSimdActiveLaneCount;

    FORCEINLINE RayStats()
    {
        reset();
    }

    FORCEINLINE void reset()
    {
        rayCount = 0;
        nodeCount = 0;
        primCount = 0;

        shadeSimdBatchCount = 0;
        shadeSimdActiveLaneCount = 0;
    }

    RayStats& operator +=(const RayStats& other)
    {
        rayCount += other.rayCount;
        nodeCount += other.nodeCount;
        primCount += other.primCount;

        shadeSimdBatchCount += other.shadeSimdBatchCount;
        shadeSimdActiveLaneCount += other.shadeSimdActiveLaneCount;

        return *this;
    }
};

} // namespace prt
