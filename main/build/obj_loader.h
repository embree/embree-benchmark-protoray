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

#include "sys/common.h"
#include "sys/array.h"
#include "sys/string.h"
#include "geometry/triangle_mesh.h"
#include "geometry/triangle_soup.h"

namespace prt {

class ObjLoader : Uncopyable
{
private:
    struct ObjVertex
    {
        int p; // position
        int t; // texcoord
        int n; // normal
    };

    struct ObjTriangle
    {
        ObjVertex v[3];
        int matId;
    };

    char* buffer;
    int bufferSize;
    Array<Vec3f> positionList;
    Array<Vec3f> normalList;
    Array<Vec2f> texcoordList;
    Array<ObjTriangle> triangleList;
    Array<std::string> materialNameList;
    Array<ObjVertex> faceVertexList; // temporary
    int currentMaterialId;
    int degenerateTriangleCount;

public:
    ObjLoader();
    ~ObjLoader();

    bool load(const std::string& filename, TriangleSoup& soup);

private:
    bool parseObj(const std::string& filename);
    bool parsePosition();
    bool parseTexcoord();
    bool parseNormal();
    bool parseFace();
    bool parseUseMtl();
    int convertIndex(int i, int size);

    bool makeSoup(TriangleSoup& soup);
    void cleanup();
};

} // namespace prt
