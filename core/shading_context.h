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

#include "sys/memory.h"
#include "ray.h"
#include "ray_simd.h"

namespace prt {

const int shadingContextDataSize = 4096;

class ShadingContext
{
public:
    Vec3f p;         // position
    Vec3f U;         // shading tangent
    Vec3f V;         // shading bitangent
    Vec3f N;         // shading normal
    Vec3f Ng;        // geometric normal
    Vec2f uv;        // texture coords
    bool backfacing; // is the geometry backfacing?
    float eps;       // intersection epsilon

private:
    // Dynamically allocated data
    char* ptr;
    ALIGNED_SIMD char data[shadingContextDataSize];

public:
    FORCEINLINE void begin()
    {
        ptr = data;
    }

    template <class T, class... Args>
    FORCEINLINE T* make(Args&&... args)
    {
        T* obj = (T*)ptr;
        new (obj) T(std::forward<Args>(args)...);
        ptr += sizeof(T);
        return obj;
    }

    FORCEINLINE Basis3f getBasis() const
    {
        return Basis3f(U, V, N);
    }

    FORCEINLINE Vec3f getN() const
    {
        return N;
    }
};

class ShadingContextSimd
{
public:
    Vec3vf p;         // position
    Vec3vf U;         // shading tangent
    Vec3vf V;         // shading bitangent
    Vec3vf N;         // shading normal
    Vec3vf Ng;        // geometric normal
    Vec2vf uv;        // texture coords
    vbool backfacing; // is the geometry backfacing?
    vfloat eps;       // intersection epsilon

private:
    // Dynamically allocated data
    char* ptr;
    ALIGNED_SIMD char data[shadingContextDataSize];

public:
    FORCEINLINE void begin()
    {
        ptr = data;
    }

    template <class T, class... Args>
    FORCEINLINE T* make(Args&&... args)
    {
        T* obj = (T*)ptr;
        new (obj) T(std::forward<Args>(args)...);
        ptr += sizeof(T);
        return obj;
    }

    FORCEINLINE Basis3vf getBasis() const
    {
        return Basis3vf(U, V, N);
    }

    FORCEINLINE Vec3vf getN() const
    {
        return N;
    }
};

class SimpleShadingContext
{
public:
    Vec3f p;   // position
    Vec3f Ng;  // geometric normal
    float eps; // intersection epsilon

    FORCEINLINE Basis3f getBasis() const
    {
        return makeBasis(Ng);
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

    FORCEINLINE Basis3vf getBasis() const
    {
        return makeBasis(Ng);
    }

    FORCEINLINE Vec3vf getN() const
    {
        return Ng;
    }
};

} // namespace prt
