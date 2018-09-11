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

#include "sys/timer_cuda.h"
#include "sys/logging.h"
#include "optix_intersector_stream_cuda.h"

namespace prt {

OptixIntersectorStreamCuda::OptixIntersectorStreamCuda(const TriangleMeshCuda& mesh, const Props& props, Props& stats)
{
    TimerCuda timer;
    bool isBenchmark = props.exists("benchmark");
    int buildCount = props.get("buildCount", isBenchmark ? 12 : 1);
    int buildWarmup = props.get("buildWarmup", buildCount / 6);

    // Create context
    Log() << "Creating OptiX Prime context";
    Log() << optix::prime::getVersionString();
    context = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);

    // Use only the first CUDA device
    std::vector<unsigned> devNums = {0};
    context->setCudaDeviceNumbers(devNums);

    // Create model
    model = context->createModel();

    // Build
    if (buildCount == 1)
        Log() << "Building acceleration structure";
    else
        Log() << "Building acceleration structure (" << buildCount << "x)";

    double buildTimeSum = 0.0;
    for (int buildIndex = 0; buildIndex < buildCount; ++buildIndex)
    {
        model->setTriangles(mesh.triangleCount, RTP_BUFFER_TYPE_CUDA_LINEAR, mesh.indices,
                            mesh.vertexCount,   RTP_BUFFER_TYPE_CUDA_LINEAR, mesh.positions);

        timer.start();
        model->update(0);
        double buildTime = timer.stop();
        if (buildCount == 1 || buildIndex >= buildWarmup)
            buildTimeSum += buildTime;
    }

    // Stats
    double buildTimeAvg = (buildCount == 1) ? buildTimeSum : (buildTimeSum / double(max(buildCount-buildWarmup, 0)));
    double buildMsAvg = buildTimeAvg * 1000.0;
    double buildMprimAvg = double(mesh.triangleCount) / 1000000.0 / buildTimeAvg;

    stats.set("buildMs", buildMsAvg);
    stats.set("buildMprim", buildMprimAvg);

    Log() << "Build time: " << buildMsAvg << " ms";
    Log() << "Build speed: " << buildMprimAvg << " Mprim/s";

    // Create queries
    closestQuery = model->createQuery(RTP_QUERY_TYPE_CLOSEST);
    anyQuery = model->createQuery(RTP_QUERY_TYPE_ANY);
}

void OptixIntersectorStreamCuda::intersect(RayCuda* rays, HitCuda* hits, int count)
{
    closestQuery->setRays(count, RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_CUDA_LINEAR, rays);
    closestQuery->setHits(count, RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V, RTP_BUFFER_TYPE_CUDA_LINEAR, hits);
    closestQuery->execute(0);
}

void OptixIntersectorStreamCuda::occluded(RayCuda* rays, AnyHitCuda* hits, int count)
{
    anyQuery->setRays(count, RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_CUDA_LINEAR, rays);
    anyQuery->setHits(count, RTP_BUFFER_FORMAT_HIT_T, RTP_BUFFER_TYPE_CUDA_LINEAR, hits);
    anyQuery->execute(0);
}

} // namespace prt
