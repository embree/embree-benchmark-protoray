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

#include "sys/memory.h"
#include "ray.h"
#include "ray_simd.h"

namespace prt {

class ShadingContext
{
public:
    Vec3f p;         // position
    Basis3f f;       // shading frame
    Vec3f Ng;        // geometric normal
    Vec2f uv;        // texture coords
    bool backfacing; // is the geometry backfacing?
    float eps;       // intersection epsilon

public:
    FORCEINLINE const Basis3f& getFrame() const
    {
        return f;
    }

    FORCEINLINE const Vec3f& getN() const
    {
        return f.N;
    }
};

class ShadingContextSimd
{
public:
    Vec3vf p;         // position
    Basis3vf f;       // shading frame
    Vec3vf Ng;        // geometric normal
    Vec2vf uv;        // texture coords
    vbool backfacing; // is the geometry backfacing?
    vfloat eps;       // intersection epsilon

public:
    FORCEINLINE const Basis3vf& getFrame() const
    {
        return f;
    }

    FORCEINLINE const Vec3vf& getN() const
    {
        return f.N;
    }
};

class SimpleShadingContext
{
public:
    Vec3f p;   // position
    Vec3f Ng;  // geometric normal
    float eps; // intersection epsilon

    FORCEINLINE Basis3f getFrame() const
    {
        return makeFrame(Ng);
    }

    FORCEINLINE Vec3f getN() const
    {
        return Ng;
    }
};

class SimpleShadingContextSimd
{
public:
    Vec3vf p;   // position
    Vec3vf Ng;  // geometric normal
    vfloat eps; // intersection epsilon

    FORCEINLINE Basis3vf getFrame() const
    {
        return makeFrame(Ng);
    }

    FORCEINLINE Vec3vf getN() const
    {
        return Ng;
    }
};

} // namespace prt
