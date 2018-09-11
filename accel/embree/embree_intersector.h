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

#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>
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
    RTCGeometry geom;

public:
    EmbreeIntersector(ref<const TriangleMesh> mesh, const Props& props, Props& stats);
    virtual ~EmbreeIntersector();

protected:
    FORCEINLINE void initIntersectContext(RTCIntersectContext& context, RayHint hint)
    {
        rtcInitIntersectContext(&context);

        if (hint == rayHintCoherent)
            context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
        else
            context.flags = RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;
    }

    FORCEINLINE void initRay(const Ray& ray, RTCRay& eray)
    {
        eray.org_x = ray.org.x;
        eray.org_y = ray.org.y;
        eray.org_z = ray.org.z;

        eray.dir_x = ray.dir.x;
        eray.dir_y = ray.dir.y;
        eray.dir_z = ray.dir.z;

        eray.tnear = 0.0f;
        eray.tfar = ray.far;

        eray.mask = -1;
        eray.time = 0.0f;
    }

    FORCEINLINE void initHit(RTCHit& ehit)
    {
        ehit.geomID = RTC_INVALID_GEOMETRY_ID;
    }

    template <class RTCRayT>
    FORCEINLINE void initRay(const RaySimd& ray, RTCRayT& eray)
    {
        store(eray.org_x, ray.org.x);
        store(eray.org_y, ray.org.y);
        store(eray.org_z, ray.org.z);

        store(eray.dir_x, ray.dir.x);
        store(eray.dir_y, ray.dir.y);
        store(eray.dir_z, ray.dir.z);

        store(eray.tnear, vfloat(zero));
        store(eray.tfar, ray.far);

        store((int*)eray.mask, vint(-1));
        store(eray.time, vfloat(zero));
    }

    template <class RTCHitT>
    FORCEINLINE void initHit(RTCHitT& ehit)
    {
        store((int*)ehit.geomID, vint(RTC_INVALID_GEOMETRY_ID));
    }

    FORCEINLINE void initRay(const RaySimd& ray, int i, RTCRay8& eray)
    {
        store(eray.org_x, load<8>(&ray.org.x[i]));
        store(eray.org_y, load<8>(&ray.org.y[i]));
        store(eray.org_z, load<8>(&ray.org.z[i]));

        store(eray.dir_x, load<8>(&ray.dir.x[i]));
        store(eray.dir_y, load<8>(&ray.dir.y[i]));
        store(eray.dir_z, load<8>(&ray.dir.z[i]));

        store(eray.tnear, vfloat8(zero));
        store(eray.tfar, load<8>(&ray.far[i]));

        store((int*)eray.mask, vint8(-1));
        store(eray.time, vfloat8(zero));
    }

    FORCEINLINE void initHit(RTCHit8& ehit)
    {
        store((int*)ehit.geomID, vint8(RTC_INVALID_GEOMETRY_ID));
    }

    FORCEINLINE void initRay(const RaySimd& ray, int i, RTCRay& eray)
    {
        eray.org_x = ray.org.x[i];
        eray.org_y = ray.org.y[i];
        eray.org_z = ray.org.z[i];

        eray.dir_x = ray.dir.x[i];
        eray.dir_y = ray.dir.y[i];
        eray.dir_z = ray.dir.z[i];

        eray.tnear = 0.0f;
        eray.tfar = ray.far[i];

        eray.mask = -1;
        eray.time = 0.0f;
    }

    template <int streamSize>
    FORCEINLINE void initRay(const RayStream<streamSize>& rays, int i, RTCRay& eray)
    {
        eray.org_x = rays.org.x[i];
        eray.org_y = rays.org.y[i];
        eray.org_z = rays.org.z[i];

        eray.dir_x = rays.dir.x[i];
        eray.dir_y = rays.dir.y[i];
        eray.dir_z = rays.dir.z[i];

        eray.tnear = 0.0f;
        eray.tfar = rays.far[i];

        eray.mask = -1;
        eray.time = 0.0f;
    }
};

} // namespace prt
