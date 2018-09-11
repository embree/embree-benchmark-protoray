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

#include "math/math.cuh"
#include "core/ray_cuda.cuh"
#include "core/shading_context_cuda.cuh"
#include "triangle_mesh_cuda.h"

namespace prt {

CUDA_DEV_FORCEINLINE float3 getPosition(const TriangleMeshCuda& mesh, int i)
{
    return mesh.positions[i];
}

CUDA_DEV_FORCEINLINE float3 getNormal(const TriangleMeshCuda& mesh, int i)
{
    return mesh.normals[i];
}

CUDA_DEV_FORCEINLINE float2 getTexcoord(const TriangleMeshCuda& mesh, int i)
{
    return mesh.texcoords[i];
}

CUDA_DEV_FORCEINLINE void postIntersect(const TriangleMeshCuda& mesh, const RayCuda& ray, const HitCuda& hit, ShadingContextCuda& ctx)
{
    int3 tri = mesh.indices[hit.primId];

    // Get the barycentric coordinates
    float b0 = hit.u;
    float b1 = hit.v;
    float b2 = 1.0f - b0 - b1;

    // Compute the hit point
    ctx.p = ray.getHitPoint(hit, ctx.eps);

    // Compute the normals
    float3 p0 = getPosition(mesh, tri.x);
    float3 p1 = getPosition(mesh, tri.y);
    float3 p2 = getPosition(mesh, tri.z);

    float3 dp1 = p1 - p0;
    float3 dp2 = p2 - p0;

    ctx.Ng = normalize(cross(dp1, dp2));

    if (mesh.normals)
    {
        float3 n0 = getNormal(mesh, tri.x);
        float3 n1 = getNormal(mesh, tri.y);
        float3 n2 = getNormal(mesh, tri.z);

        ctx.f.N = normalize(b0*n0 + b1*n1 + b2*n2);
        if (dot(ctx.f.N, ctx.Ng) < 0.0f)
            ctx.f.N = -ctx.f.N;
    }
    else
    {
        ctx.f.N = ctx.Ng;
    }

    ctx.backfacing = dot(ctx.Ng, ray.dir) > 0.0f;
    if (ctx.backfacing)
    {
        ctx.Ng = -ctx.Ng;
        ctx.f.N = -ctx.f.N;
    }

    // Compute the UVs
    if (mesh.texcoords)
    {
        float2 uv0 = getTexcoord(mesh, tri.x);
        float2 uv1 = getTexcoord(mesh, tri.y);
        float2 uv2 = getTexcoord(mesh, tri.z);

        ctx.uv = b0*uv0 + b1*uv1 + b2*uv2;

        // Compute partial derivatives
        float du1 = uv1.x - uv0.x;
        float du2 = uv2.x - uv0.x;
        float dv1 = uv1.y - uv0.y;
        float dv2 = uv2.y - uv0.y;

        float det = du1 * dv2 - dv1 * du2;
        if (det != 0.0f)
        {
            float invDet = rcp(det);
            float3 dpdu = (dv2 * dp1 - dv1 * dp2) * invDet;
            //float3 dpdv = (du1 * dp2 - du2 * dp1) * invDet;

            // Compute frame
            ctx.f.U = normalize(dpdu);
            ctx.f.V = cross(ctx.f.N, ctx.f.U);
            if (lengthSqr(ctx.f.V) > 0.0f)
            {
                ctx.f.V = normalize(ctx.f.V);
                ctx.f.U = cross(ctx.f.V, ctx.f.N);
            }
            else
            {
                makeFrame(ctx.f.U, ctx.f.V, ctx.f.N);
            }
        }
        else
        {
            makeFrame(ctx.f.U, ctx.f.V, ctx.f.N);
        }
    }
    else
    {
        ctx.uv = make_float2(hit.u, hit.v);
        makeFrame(ctx.f.U, ctx.f.V, ctx.f.N);
    }
}

CUDA_DEV_FORCEINLINE void postIntersect(const TriangleMeshCuda& mesh, const RayCuda& ray, const HitCuda& hit, SimpleShadingContextCuda& ctx)
{
    int3 tri = mesh.indices[hit.primId];

    // Compute the hit point
    ctx.p = ray.getHitPoint(hit, ctx.eps);

    // Compute the normals
    float3 p0 = getPosition(mesh, tri.x);
    float3 p1 = getPosition(mesh, tri.y);
    float3 p2 = getPosition(mesh, tri.z);

    float3 dp1 = p1 - p0;
    float3 dp2 = p2 - p0;

    ctx.Ng = normalize(cross(dp1, dp2));

    bool backfacing = dot(ctx.Ng, ray.dir) > 0.0f;
    if (backfacing)
        ctx.Ng = -ctx.Ng;
}

} // namespace prt
