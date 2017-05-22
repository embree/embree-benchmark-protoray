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

#include "sys/logging.h"
#include "sys/blob.h"
#include "sys/filesystem.h"
#include "geometry/triangle_mesh.h"
#include "scene.h"

namespace prt {

Scene::Scene(const std::string& path, const Props& props)
{
    this->path = path;
    std::string pathBase = getFilenameBase(path);

    // Load the mesh
    ref<TriangleMesh> mesh = makeRef<TriangleMesh>();
    loadBlob(path, *mesh);
    shape = mesh;

    // Load the materials
    materialNames = shape->getMaterialNames();
}

} // namespace prt
