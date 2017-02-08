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

#include "math/box3.h"
#include "vertex.h"

namespace prt {

struct Triangle
{
    Vec3f v[3];

    Triangle() {}

    Triangle(const Vec3f& v0, const Vec3f& v1, const Vec3f& v2)
    {
        v[0] = v0;
        v[1] = v1;
        v[2] = v2;
    }

    Box3f getBounds() const
    {
        return Box3f(min(min(v[0], v[1]), v[2]),
                     max(max(v[0], v[1]), v[2]));
    }

    Vec3f getCenter() const
    {
        return (v[0] + v[1] + v[2]) * (1.0f / 3.0f);
    }

    Vec3f getNormal() const
    {
        return cross(v[1] - v[0], v[2] - v[0]);
    }
};

struct FatTriangle
{
    FatVertex v[3];
    int matId;

    Vec3f getCenter() const
    {
        return (v[0].pos + v[1].pos + v[2].pos) * (1.0f / 3.0f);
    }

    Box3f getBounds() const
    {
        return Box3f(min(min(v[0].pos, v[1].pos), v[2].pos),
                     max(max(v[0].pos, v[1].pos), v[2].pos));
    }

    Vec3f getNormal() const
    {
        return cross(v[1].pos - v[0].pos, v[2].pos - v[0].pos);
    }
};

struct FatIndexedTriangle
{
    int v[3];
    int matId;
};

} // namespace prt
