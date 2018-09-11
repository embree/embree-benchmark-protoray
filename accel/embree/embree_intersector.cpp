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

#include "sys/timer.h"
#include "embree_intersector.h"

namespace prt {

EmbreeIntersector::EmbreeIntersector(ref<const TriangleMesh> mesh, const Props& props, Props& stats)
    : mesh(mesh)
{
    bool isHighQuality = !props.exists("no-sbvh");
    bool isBenchmark = props.exists("benchmark");
    int buildCount = props.get("buildCount", isBenchmark ? 12 : 1);
    int buildWarmup = props.get("buildWarmup", buildCount / 6);

    std::string deviceCfg = props.get("embree", "start_threads=1,set_affinity=1");
    deviceCfg = props.get("rtcore", deviceCfg);

    if (buildCount > 1)
    {
        std::string deviceCfg2 = "tri_accel=bvh8.triangle4,tri_builder=";
        if (isHighQuality)
            deviceCfg2 += "sah_fast_spatial";
        else
            deviceCfg2 += "sah";

        if (deviceCfg.empty())
            deviceCfg = deviceCfg2;
        else
            deviceCfg = deviceCfg2 + "," + deviceCfg;
    }

    // Create device
    Log() << "Creating Embree device: " << deviceCfg;
    device = rtcNewDevice(deviceCfg.c_str());

    // Create scene
    RTCSceneFlags sceneFlags = (buildCount == 1 ? RTC_SCENE_FLAG_NONE : RTC_SCENE_FLAG_DYNAMIC);

    scene = rtcNewScene(device);
    rtcSetSceneFlags(scene, sceneFlags);
    if (isHighQuality)
        rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);

    // Build
    if (buildCount == 1)
        Log() << "Building acceleration structure";
    else
        Log() << "Building acceleration structure (" << buildCount << "x)";

    geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    rtcAttachGeometry(scene, geom);

    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, mesh->indices.getData(), 0, sizeof(Vec3i), mesh->indices.getSize());

    Vec3f* vertices = (Vec3f*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vec3f), mesh->vertexCount);
    for (int i = 0; i < mesh->vertexCount; ++i)
        vertices[i] = mesh->getPosition(i);

    Timer timer;
    double buildTimeSum = 0.0;

    for (int buildIndex = 0; buildIndex < buildCount; ++buildIndex)
    {
        timer.reset();
        rtcCommitGeometry(geom);
        rtcCommitScene(scene);
        double buildTime = timer.query();
        if (buildCount == 1 || buildIndex >= buildWarmup)
            buildTimeSum += buildTime;
    }

    // Stats
    double buildTimeAvg = (buildCount == 1) ? buildTimeSum : (buildTimeSum / double(max(buildCount-buildWarmup, 0)));
    double buildMsAvg = buildTimeAvg * 1000.0;
    double buildMprimAvg = double(mesh->getPrimCount()) / 1000000.0 / buildTimeAvg;

    stats.set("buildMs", buildMsAvg);
    stats.set("buildMprim", buildMprimAvg);

    Log() << "Build time: " << buildMsAvg << " ms";
    Log() << "Build speed: " << buildMprimAvg << " Mprim/s";
}

EmbreeIntersector::~EmbreeIntersector()
{
    rtcReleaseGeometry(geom);
    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
}

} // namespace prt
