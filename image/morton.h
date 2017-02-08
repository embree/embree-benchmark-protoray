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

#include "math/math.h"
#include "math/vec2.h"

namespace prt {

extern const int mortonTableX[];
extern const int mortonTableY[];

template <int tileSizeX>
FORCEINLINE Vec2i getMorton8x8(int i)
{
    static const int logTileSizeX = bitScan(tileSizeX);
    int mortonId = i & 63;
    return Vec2i(mortonTableX[mortonId] + ((i >> 3) & (tileSizeX-8)),
                 mortonTableY[mortonId] + ((i >> (logTileSizeX+3)) << 3));
}

template <int tileSizeX>
FORCEINLINE Vec2vi getMorton8x8(vint i)
{
    static const int logTileSizeX = bitScan(tileSizeX);
    vint mortonId = i & 63;
    return Vec2vi(gather(mortonTableX, mortonId) + ((i >> 3) & (tileSizeX-8)),
                  gather(mortonTableY, mortonId) + ((i >> (logTileSizeX+3)) << 3));
}

template <int tileSizeX>
FORCEINLINE Vec2vi getMorton8x8Step(int i)
{
    static const int logTileSizeX = bitScan(tileSizeX);
    int mortonId = i & 63;
    return Vec2vi(load(mortonTableX + mortonId) + ((i >> 3) & (tileSizeX-8)),
                  load(mortonTableY + mortonId) + ((i >> (logTileSizeX+3)) << 3));
}

} // namespace prt
