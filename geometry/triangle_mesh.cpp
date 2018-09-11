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

#include "sys/blob.h"
#include "triangle_mesh.h"

namespace prt {

void TriangleMesh::alloc(int triangleCount, int vertexCount, bool hasNormals, bool hasTexcoords)
{
    // Allocate indices
    indices.alloc(triangleCount);
    materialIds.alloc(triangleCount);

    // Allocate vertex attribs
    this->vertexCount = vertexCount;
    vertexStride = 3; // positions
    if (hasNormals)   vertexStride += 3;
    if (hasTexcoords) vertexStride += 2;

    vertexAttribs.alloc(vertexCount * vertexStride);

    // Setup vertex layout
    if (hasNormals && hasTexcoords)
    {
        normals = vertexAttribs.getData() + 3;
        texcoords = normals + 3;
    }
    else if (hasNormals)
    {
        normals = vertexAttribs.getData() + 3;
        texcoords = 0;
    }
    else if (hasTexcoords)
    {
        texcoords = vertexAttribs.getData() + 3;
        normals = 0;
    }
    else
    {
        normals = 0;
        texcoords = 0;
    }
}

void TriangleMesh::postIntersect(const Ray& ray, const Hit& hit, ShadingContext& ctx) const
{
    prefetchL1(&indices[hit.primId]);
    Vec3i tri = indices[hit.primId];

    prefetchVertexL1(tri[0]);
    prefetchVertexL1(tri[1]);
    prefetchVertexL1(tri[2]);

    // Get the barycentric coordinates
    float b1 = hit.u;
    float b2 = hit.v;
    float b0 = 1.0f - b1 - b2;

    // Compute the hit point
    ctx.p = ray.getHitPoint(ctx.eps);

    // Compute the normals
    Vec3f p0 = getPosition(tri[0]);
    Vec3f p1 = getPosition(tri[1]);
    Vec3f p2 = getPosition(tri[2]);

    Vec3f dp1 = p1 - p0;
    Vec3f dp2 = p2 - p0;

    ctx.Ng = normalize(cross(dp1, dp2));

    if (normals)
    {
        Vec3f n0 = getNormal(tri[0]);
        Vec3f n1 = getNormal(tri[1]);
        Vec3f n2 = getNormal(tri[2]);

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
    if (texcoords)
    {
        Vec2f uv0 = getTexcoord(tri[0]);
        Vec2f uv1 = getTexcoord(tri[1]);
        Vec2f uv2 = getTexcoord(tri[2]);

        ctx.uv = b0*uv0 + b1*uv1 + b2*uv2;

        // Compute partial derivatives
        float du1 = uv1.x - uv0.x;
        float du2 = uv2.x - uv0.x;
        float dv1 = uv1.y - uv0.y;
        float dv2 = uv2.y - uv0.y;

        float det = du1 * dv2 - dv1 * du2;
        if (LIKELY(det != 0.0f))
        {
            float invDet = rcp(det);
            Vec3f dpdu = (dv2 * dp1 - dv1 * dp2) * invDet;
            //Vec3f dpdv = (du1 * dp2 - du2 * dp1) * invDet;

            // Compute frame
            ctx.f.U = normalize(dpdu);
            ctx.f.V = cross(ctx.f.N, ctx.f.U);
            if (LIKELY(lengthSqr(ctx.f.V) > 0.0f))
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
        ctx.uv = Vec2f(hit.u, hit.v);
        makeFrame(ctx.f.U, ctx.f.V, ctx.f.N);
    }
}

void TriangleMesh::postIntersect(vbool m, const RaySimd& ray, const HitSimd& hit, ShadingContextSimd& ctx) const
{
    prefetchGatherL1(m, &indices[0][0], hit.primId*3);

    Vec3vi tri;
    tri[0] = gather(m, &indices[0][0], hit.primId*3);
    tri[1] = gather(m, &indices[0][1], hit.primId*3);
    tri[2] = gather(m, &indices[0][2], hit.primId*3);

    prefetchVertexL1(m, tri[0]);
    prefetchVertexL1(m, tri[1]);
    prefetchVertexL1(m, tri[2]);

    // Get the barycentric coordinates
    vfloat b1 = hit.u;
    vfloat b2 = hit.v;
    vfloat b0 = 1.0f - b1 - b2;

    // Compute the hit point
    ctx.p = ray.getHitPoint(ctx.eps);

    // Compute the normals
    Vec3vf p0 = getPosition(m, tri[0]);
    Vec3vf p1 = getPosition(m, tri[1]);
    Vec3vf p2 = getPosition(m, tri[2]);

    Vec3vf dp1 = p1 - p0;
    Vec3vf dp2 = p2 - p0;

    ctx.Ng = normalize(cross(dp1, dp2));

    if (normals)
    {
        Vec3vf n0 = getNormal(m, tri[0]);
        Vec3vf n1 = getNormal(m, tri[1]);
        Vec3vf n2 = getNormal(m, tri[2]);

        ctx.f.N = normalize(b0*n0 + b1*n1 + b2*n2);
        set(dot(ctx.f.N, ctx.Ng) < 0.0f, ctx.f.N, -ctx.f.N);
    }
    else
    {
        ctx.f.N = ctx.Ng;
    }

    ctx.backfacing = dot(ctx.Ng, ray.dir) > 0.0f;
    set(ctx.backfacing, ctx.Ng, -ctx.Ng);
    set(ctx.backfacing, ctx.f.N, -ctx.f.N);

    if (texcoords)
    {
        Vec2vf uv0 = getTexcoord(m, tri[0]);
        Vec2vf uv1 = getTexcoord(m, tri[1]);
        Vec2vf uv2 = getTexcoord(m, tri[2]);

        ctx.uv = b0*uv0 + b1*uv1 + b2*uv2;

        // Compute partial derivatives
        vfloat du1 = uv1.x - uv0.x;
        vfloat du2 = uv2.x - uv0.x;
        vfloat dv1 = uv1.y - uv0.y;
        vfloat dv2 = uv2.y - uv0.y;

        vfloat det = du1 * dv2 - dv1 * du2;
        vbool isDetZero = det == 0.0f;

        vfloat invDet = rcp(det);
        Vec3vf dpdu = (dv2 * dp1 - dv1 * dp2) * invDet;
        //Vec3vf dpdv = (du1 * dp2 - du2 * dp1) * invDet;

        // Compute frame
        ctx.f.U = normalize(dpdu);
        ctx.f.V = cross(ctx.f.N, ctx.f.U);
        isDetZero |= lengthSqr(ctx.f.V) == 0.0f;
        ctx.f.V = normalize(ctx.f.V);
        ctx.f.U = cross(ctx.f.V, ctx.f.N);

        isDetZero &= m;
        if (any(isDetZero))
        {
            Vec3vf U2, V2;
            makeFrame(U2, V2, ctx.f.N);
            set(isDetZero, ctx.f.U, U2);
            set(isDetZero, ctx.f.V, V2);
        }
    }
    else
    {
        ctx.uv = Vec2vf(hit.u, hit.v);
        makeFrame(ctx.f.U, ctx.f.V, ctx.f.N);
    }
}

void TriangleMesh::postIntersect(const Ray& ray, const Hit& hit, SimpleShadingContext& ctx) const
{
    prefetchL1(&indices[hit.primId]);
    Vec3i tri = indices[hit.primId];

    prefetchVertexL1(tri[0]);
    prefetchVertexL1(tri[1]);
    prefetchVertexL1(tri[2]);

    // Compute the hit point
    ctx.p = ray.getHitPoint(ctx.eps);

    // Compute the normals
    Vec3f p0 = getPosition(tri[0]);
    Vec3f p1 = getPosition(tri[1]);
    Vec3f p2 = getPosition(tri[2]);

    Vec3f dp1 = p1 - p0;
    Vec3f dp2 = p2 - p0;

    ctx.Ng = normalize(cross(dp1, dp2));

    bool backfacing = dot(ctx.Ng, ray.dir) > 0.0f;
    if (backfacing)
        ctx.Ng = -ctx.Ng;
}

void TriangleMesh::postIntersect(vbool m, const RaySimd& ray, const HitSimd& hit, SimpleShadingContextSimd& ctx) const
{
    prefetchGatherL1(m, &indices[0][0], hit.primId*3);

    Vec3vi tri;
    tri[0] = gather(m, &indices[0][0], hit.primId*3);
    tri[1] = gather(m, &indices[0][1], hit.primId*3);
    tri[2] = gather(m, &indices[0][2], hit.primId*3);

    prefetchVertexL1(m, tri[0]);
    prefetchVertexL1(m, tri[1]);
    prefetchVertexL1(m, tri[2]);

    // Compute the hit point
    ctx.p = ray.getHitPoint(ctx.eps);

    // Compute the normals
    Vec3vf p0 = getPosition(m, tri[0]);
    Vec3vf p1 = getPosition(m, tri[1]);
    Vec3vf p2 = getPosition(m, tri[2]);

    Vec3vf dp1 = p1 - p0;
    Vec3vf dp2 = p2 - p0;

    ctx.Ng = normalize(cross(dp1, dp2));

    vbool backfacing = dot(ctx.Ng, ray.dir) > 0.0f;
    set(backfacing, ctx.Ng, -ctx.Ng);
}

Stream& operator >>(Stream& ism, TriangleMesh& mesh)
{
    ism >> mesh.bounds;

    int triangleCount, vertexCount;
    bool hasNormals, hasTexcoords;
    ism >> triangleCount >> vertexCount >> hasNormals >> hasTexcoords;

    mesh.alloc(triangleCount, vertexCount, hasNormals, hasTexcoords);
    ism.readFull(mesh.indices.getData(), mesh.indices.getSize() * sizeof(Vec3i));
    ism.readFull(mesh.materialIds.getData(), mesh.materialIds.getSize() * sizeof(int));
    ism.readFull(mesh.vertexAttribs.getData(), mesh.vertexAttribs.getSize() * sizeof(float));

    ism >> mesh.materialNames;
    return ism;
}

Stream& operator <<(Stream& osm, const TriangleMesh& mesh)
{
    osm << mesh.bounds;

    int triangleCount = mesh.indices.getSize();
    bool hasNormals = mesh.normals;
    bool hasTexcoords = mesh.texcoords;
    osm << triangleCount << mesh.vertexCount << hasNormals << hasTexcoords;

    osm.write(mesh.indices.getData(), mesh.indices.getSize() * sizeof(Vec3i));
    osm.write(mesh.materialIds.getData(), mesh.materialIds.getSize() * sizeof(int));
    osm.write(mesh.vertexAttribs.getData(), mesh.vertexAttribs.getSize() * sizeof(float));

    osm << mesh.materialNames;
    return osm;
}

} // namespace prt

