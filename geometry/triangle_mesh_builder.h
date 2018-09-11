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

#include "sys/array.h"
#include "sys/fixed_hash_map.h"
#include "math/box3.h"
#include "triangle_mesh.h"
#include "triangle_soup.h"

namespace prt {

class TriangleMeshBuilder
{
public:
    TriangleMeshBuilder(const TriangleSoup& soup, TriangleMesh& mesh);

private:
    struct Fragment
    {
        Vec3f position;
        int id;

        Fragment() {}
        Fragment(const Vec3f& position, int id) : position(position), id(id) {}
    };

    Array<FatVertex> vertices;
    Array<FatIndexedTriangle> indexedTriangles;

    Array<Fragment> vertexFragments;
    Array<Fragment> triangleFragments;

    Box3f bounds;
    bool hasNormals;
    bool hasTexcoords;
    bool hasMaterials;

    bool isReorderEnabled;

    void build(const TriangleSoup& soup, TriangleMesh& mesh);
    //void buildFast(const Array<FatTriangle>& triangles, TriangleMesh& mesh);

    void constructMeshTopology(const Array<FatTriangle>& triangles);
    void reorderFragments();
    void reorderFragments(Fragment* vertexBegin, Fragment* vertexEnd, Fragment* triangleBegin, Fragment* triangleEnd);
    void storeMesh(TriangleMesh& mesh);
    void cleanup();
};

} // namespace prt
