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
#include "sys/common.h"
#include "geometry/triangle.h"

namespace prt {

struct TriangleSoup
{
    Array<FatTriangle> triangles;
    Array<std::string> materialNames;
};

bool loadTriangleSoup(const std::string& filename, TriangleSoup& soup);
bool saveTriangleSoup(const std::string& filename, const TriangleSoup& soup);

} // namespace prt
