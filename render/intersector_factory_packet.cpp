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
#include "accel/embree/embree_intersector_packet.h"
#endif

#include "intersector_factory_packet.h"

namespace prt {

ref<IntersectorPacket> IntersectorFactoryPacket::make(const ref<const Scene>& scene, const Props& props, Props& stats)
{
    // Load the acceleration structure
    std::string defaultAccelType = "embree";

    std::string accelType = props.get("accel", defaultAccelType);
    std::string accelPath = getFilenameBase(scene->getPath()) + "." + accelType;

    const std::string defaultIsectType = "packet";
    std::string isectType = props.get("isect", defaultIsectType);

    Log() << "Acceleration structure: " << accelType;
    Log() << "Intersector: " << isectType;

#ifdef EMBREE_SUPPORT
    if (accelType == "embree")
    {
        if (isectType == "single")
            return makeRef<EmbreeIntersectorSinglePacket>(scene->shape, props, stats);
        if (isectType == "packet")
            return makeRef<EmbreeIntersectorPacket>(scene->shape, props, stats);
        if (isectType == "packet8")
            return makeRef<EmbreeIntersectorPacket8>(scene->shape, props, stats);

        throw std::invalid_argument("invalid intersector type");
    }
#endif

    throw std::invalid_argument("invalid acceleration structure type");
}

} // namespace prt
