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

#include "sys/ref.h"
#include "sys/array.h"
#include "sys/props.h"
#include "geometry/shape.h"
#include "geometry/triangle_mesh.h"

namespace prt {

class Scene
{
//private:
public:
    //ref<Shape> shape;                     // scene shape
    ref<TriangleMesh> shape;

    Array<std::string> materialNames;       // material names of the shape
    std::string path;                       // path to the scene

public:
    Scene(const std::string& path, const Props& props);

    std::string getPath() const
    {
        return path;
    }

    // Shape
    // -----

    FORCEINLINE void postIntersect(const Ray& ray, const Hit& hit, ShadingContext& ctx) const
    {
        shape->postIntersect(ray, hit, ctx);
    }

    FORCEINLINE void postIntersect(vbool m, const RaySimd& ray, const HitSimd& hit, ShadingContextSimd& ctx) const
    {
        shape->postIntersect(m, ray, hit, ctx);
    }

    FORCEINLINE void postIntersect(const Ray& ray, const Hit& hit, SimpleShadingContext& ctx) const
    {
        shape->postIntersect(ray, hit, ctx);
    }

    FORCEINLINE void postIntersect(vbool m, const RaySimd& ray, const HitSimd& hit, SimpleShadingContextSimd& ctx) const
    {
        shape->postIntersect(m, ray, hit, ctx);
    }

    FORCEINLINE Box3f getBounds() const
    {
        return shape->getBounds();
    }

    FORCEINLINE int getMaterialId(int primId) const
    {
        return shape->materialIds[primId];
    }

    FORCEINLINE vint getMaterialId(vbool m, vint primId) const
    {
        return gather(m, shape->materialIds.getData(), primId);
    }

    FORCEINLINE std::string getMaterialName(int matId) const
    {
        return materialNames[matId];
    }
};

} // namespace prt
