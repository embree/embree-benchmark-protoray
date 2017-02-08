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

#include "geometry/triangle_mesh.h"
#include "triangle_soup.h"

namespace prt {

bool loadTriangleSoup(const std::string& filename, TriangleSoup& soup)
{
    FILE* file = fopen(filename.c_str(), "rb");
    if (file == 0)
        return false;

    fseek(file, 0, SEEK_END);
    int triangleCount = ftell(file) / sizeof(FatTriangle);
    fseek(file, 0, SEEK_SET);
    soup.triangles.alloc(triangleCount);
    fread(soup.triangles.getData(), sizeof(FatTriangle), triangleCount, file);
    fclose(file);

    soup.materialNames.clear();
    soup.materialNames.pushBack("Default");
    return true;
}

bool saveTriangleSoup(const std::string& filename, const TriangleSoup& soup)
{
    FILE* file = fopen(filename.c_str(), "wb");
    if (file == 0)
        return false;

    fwrite(soup.triangles.getData(), sizeof(FatTriangle), soup.triangles.getSize(), file);
    fclose(file);
    return true;
}

} // namespace prt

