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

#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include "sys/ref.h"
#include "sys/props.h"
#include "sys/logging.h"
#include "geometry/triangle_mesh.h"
#include "core/intersector_single.h"
#include "core/intersector_packet.h"
#include "core/intersector_stream.h"

namespace prt {

class EmbreeIntersector
{
protected:
    ref<const TriangleMesh> mesh;
    RTCDevice device;
    RTCScene scene;
    unsigned int geomID;

public:
    EmbreeIntersector(ref<const TriangleMesh> mesh, const Props& props, Props& stats);
    virtual ~EmbreeIntersector();

protected:
    FORCEINLINE void initIntersectContext(RTCIntersectContext& context, RayHint hint)
    {
        if (hint == rayHintCoherent)
            context.flags = RTC_INTERSECT_COHERENT;
        else
            context.flags = RTC_INTERSECT_INCOHERENT;
        context.userRayExt = 0;
    }
};

} // namespace prt
