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

#include "sys/array.h"
#include "sys/string.h"
#include "math/vec2.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "math/box3.h"
#include "triangle.h"
#include "shape.h"

namespace prt {

struct TriangleMesh : public Shape
{
    typedef Triangle Prim;

    static const int blobId = 0x04df8666;

    // Triangles
    Array<Vec3i> indices; // v0 v1 v2

    // Vertices
    int vertexCount;
    Array<float> vertexAttribs;
    int vertexStride;
    float* normals;   // Vec3f
    float* texcoords; // Vec2f

    // Materials
    Array<int> materialIds;
    Array<std::string> materialNames;

    Box3f bounds;

    void alloc(int triangleCount, int vertexCount, bool hasNormals, bool hasTexcoords);

    FORCEINLINE int getPrimCount() const { return indices.getSize(); }
    FORCEINLINE int getVertexCount() const { return vertexCount; }

    FORCEINLINE void prefetchVertexL1(int i) const { prefetchL1(vertexAttribs.getData() + i*vertexStride); }
    FORCEINLINE void prefetchVertexL1(vbool m, vint i) const { prefetchGatherL1(m, vertexAttribs.getData(), i*vertexStride); }

    FORCEINLINE Vec3f getPosition(int i) const { return *(const Vec3f*)(vertexAttribs.getData() + i*vertexStride); }
    FORCEINLINE Vec3f getNormal  (int i) const { return *(const Vec3f*)(normals   + i*vertexStride); }
    FORCEINLINE Vec2f getTexcoord(int i) const { return *(const Vec2f*)(texcoords + i*vertexStride); }

    FORCEINLINE Vec3vf getPosition(vbool m, vint i) const
    {
        return gather3(m, vertexAttribs.getData(), i*vertexStride);
    }

    FORCEINLINE Vec3vf getNormal(vbool m, vint i) const
    {
        return gather3(m, normals, i*vertexStride);
    }

    FORCEINLINE Vec2vf getTexcoord(vbool m, vint i) const
    {
        return gather2(m, texcoords, i*vertexStride);
    }

    FORCEINLINE void setPosition(int i, const Vec3f& v) { *(Vec3f*)(vertexAttribs.getData() + i*vertexStride) = v; }
    FORCEINLINE void setNormal  (int i, const Vec3f& v) { *(Vec3f*)(normals   + i*vertexStride) = v; }
    FORCEINLINE void setTexcoord(int i, const Vec2f& v) { *(Vec2f*)(texcoords + i*vertexStride) = v; }

    FORCEINLINE Triangle getPrim(int i) const
    {
        return Triangle(getPosition(indices[i][0]),
                        getPosition(indices[i][1]),
                        getPosition(indices[i][2]));
    }

    FORCEINLINE Box3f getBounds() const { return bounds; }
    FORCEINLINE const Array<std::string>& getMaterialNames() const { return materialNames; }

    void postIntersect(const Ray& ray, const Hit& hit, ShadingContext& ctx) const;
    void postIntersect(vbool m, const RaySimd& ray, const HitSimd& hit, ShadingContextSimd& ctx) const;

    void postIntersect(const Ray& ray, const Hit& hit, SimpleShadingContext& ctx) const;
    void postIntersect(vbool m, const RaySimd& ray, const HitSimd& hit, SimpleShadingContextSimd& ctx) const;
};

Stream& operator >>(Stream& ism, TriangleMesh& mesh);
Stream& operator <<(Stream& osm, const TriangleMesh& mesh);

} // namespace prt
