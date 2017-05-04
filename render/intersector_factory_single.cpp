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

#include "sys/blob.h"
#include "sys/filesystem.h"

#ifdef EMBREE_SUPPORT
#include "accel/embree/embree_intersector_single.h"
#endif

#include "intersector_factory_single.h"

namespace prt {

ref<IntersectorSingle> IntersectorFactorySingle::make(const ref<const Scene>& scene, const Props& props, Props& stats)
{
    std::string defaultAccelType = "embree";

    std::string accelType = props.get("accel", defaultAccelType);
    std::string accelPath = getFilenameBase(scene->getPath()) + "." + accelType;

    const std::string defaultIsectType = "single";
    std::string isectType = props.get("isect", defaultIsectType);

    Log() << "Acceleration structure: " << accelType;
    Log() << "Intersector: " << isectType;

    if (isectType != "single")
        throw std::invalid_argument("invalid intersector type");

#ifdef EMBREE_SUPPORT
    if (accelType == "embree")
    {
        return makeRef<EmbreeIntersectorSingle>(scene->shape, props, stats);
    }
#endif

    throw std::invalid_argument("invalid acceleration structure type");
}

} // namespace prt
