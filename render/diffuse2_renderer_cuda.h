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

#include <cuda_runtime.h>
#include "core/intersector_stream_cuda.h"
#include "renderer_cuda.h"

namespace prt {

class Diffuse2RendererCuda : public RendererCuda
{
private:
    TriangleMeshCuda mesh;
    IntersectorStreamCuda* intersector;

    HitCuda* hits;
    RayCuda* shadowRays;
    AnyHitCuda* shadowHits;
    RayCuda* rays[2];
    int* pixelIds[2];
    unsigned int* samplerStates[2];
    float* L[2];
    int* queueSize;

    int pixelCount;
    int pass;
    int maxDepth;
    bool isFast;

public:
    Diffuse2RendererCuda(const TriangleMeshCuda& mesh, IntersectorStreamCuda* intersector, int pixelCount, int maxDepth, bool isFast = false);
    ~Diffuse2RendererCuda();

    int render(const ThinLensCameraCuda& camera, const AccumBufferCuda& accumBuffer);
};

} // namespace prt
