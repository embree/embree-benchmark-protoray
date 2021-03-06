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

#include "sys/props.h"
#include "sys/blob.h"
#include "sys/filesystem.h"
#include "core/intersector_stream.h"

#ifdef EMBREE_SUPPORT
#include "accel/embree/embree_intersector_stream.h"
#endif

#include "scene.h"

namespace prt {

class IntersectorFactoryStream
{
public:
    template <int streamSize>
    static ref<IntersectorStream<streamSize>> make(const ref<const Scene>& scene, const Props& props, Props& stats)
    {
        // Load the acceleration structure
        std::string defaultAccelType = "embree";

        std::string accelType = props.get("accel", defaultAccelType);
        std::string accelPath = getFilenameBase(scene->getPath()) + "." + accelType;

        const std::string defaultIsectType = "stream";
        std::string isectType = props.get("isect", defaultIsectType);

        Log() << "Acceleration structure: " << accelType;
        Log() << "Intersector: " << isectType;

    #ifdef EMBREE_SUPPORT
        if (accelType == "embree")
        {
            if (isectType == "single")
                return makeRef<EmbreeIntersectorSingleStream<streamSize>>(scene->shape, props, stats);
            if (isectType == "packet")
                return makeRef<EmbreeIntersectorPacketStream<streamSize>>(scene->shape, props, stats);
            if (isectType == "stream" || isectType == "streamSop")
                return makeRef<EmbreeIntersectorStream<streamSize>>(scene->shape, props, stats);
            if (isectType == "streamSoa")
                return makeRef<EmbreeIntersectorStreamSoa<streamSize>>(scene->shape, props, stats);

            throw std::invalid_argument("invalid intersector type");
        }
    #endif

        throw std::invalid_argument("invalid acceleration structure type");
    }
};

} // namespace prt
