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

    std::string deviceCfg = props.get("embree", "");
    deviceCfg = props.get("rtcore", deviceCfg);

    if (buildCount > 1)
    {
        std::string deviceCfg2 = "start_threads=1,tri_accel=bvh8.triangle4,tri_builder=";
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
    RTCSceneFlags sceneFlags = (buildCount == 1 ? RTC_SCENE_STATIC : RTC_SCENE_DYNAMIC) | RTC_SCENE_INCOHERENT;
    if (isHighQuality)
        sceneFlags = sceneFlags | RTC_SCENE_HIGH_QUALITY;

    RTCAlgorithmFlags algoFlags = RTC_INTERSECT1 | RTC_INTERSECT_STREAM;
#if defined(__AVX512F__) || defined(__MIC__)
    algoFlags = algoFlags | RTC_INTERSECT16;
#else
    algoFlags = algoFlags | RTC_INTERSECT8;
#endif

    scene = rtcDeviceNewScene(device, sceneFlags, algoFlags);

    // Build
    if (buildCount == 1)
        Log() << "Building acceleration structure";
    else
        Log() << "Building acceleration structure (" << buildCount << "x)";

    Timer timer;
    double buildTimeSum = 0.0;

#if 1

    geomID = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, mesh->indices.getSize(), mesh->vertexCount, 1);
    
    Vec3i* triangles = (Vec3i*)rtcMapBuffer(scene, geomID, RTC_INDEX_BUFFER);
    for (int i = 0; i < mesh->indices.getSize(); ++i)
      triangles[i] = mesh->indices[i];
    rtcUnmapBuffer(scene, geomID, RTC_INDEX_BUFFER);
    
    Vec4f* vertices = (Vec4f*)rtcMapBuffer(scene, geomID, RTC_VERTEX_BUFFER);
    for (int i = 0; i < mesh->vertexCount; ++i)
      vertices[i] = Vec4f(mesh->getPosition(i), 0.0f);
    rtcUnmapBuffer(scene, geomID, RTC_VERTEX_BUFFER);

    for (int buildIndex = 0; buildIndex < buildCount; ++buildIndex)
    {
      timer.reset();
      rtcUpdate(scene,geomID);
      rtcCommit(scene);
      double buildTime = timer.query();
      if (buildCount == 1 || buildIndex >= buildWarmup)
        buildTimeSum += buildTime;
    }

#else
    for (int buildIndex = 0; buildIndex < buildCount; ++buildIndex)
    {
        geomID = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, mesh->indices.getSize(), mesh->vertexCount, 1);

        Vec3i* triangles = (Vec3i*)rtcMapBuffer(scene, geomID, RTC_INDEX_BUFFER);
        for (int i = 0; i < mesh->indices.getSize(); ++i)
            triangles[i] = mesh->indices[i];
        rtcUnmapBuffer(scene, geomID, RTC_INDEX_BUFFER);

        Vec4f* vertices = (Vec4f*)rtcMapBuffer(scene, geomID, RTC_VERTEX_BUFFER);
        for (int i = 0; i < mesh->vertexCount; ++i)
            vertices[i] = Vec4f(mesh->getPosition(i), 0.0f);
        rtcUnmapBuffer(scene, geomID, RTC_VERTEX_BUFFER);

        timer.reset();
        rtcCommit(scene);
        double buildTime = timer.query();
        if (buildCount == 1 || buildIndex >= buildWarmup)
            buildTimeSum += buildTime;

        if (buildIndex < buildCount-1)
            rtcDeleteGeometry(scene, geomID);
    }
#endif
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
    rtcDeleteScene(scene);
    rtcDeleteDevice(device);
}

} // namespace prt
