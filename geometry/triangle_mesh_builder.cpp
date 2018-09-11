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

#include "sys/logging.h"
#include "sys/fixed_hash_map.h"
#include "triangle_mesh_builder.h"

namespace prt {

TriangleMeshBuilder::TriangleMeshBuilder(const TriangleSoup& soup, TriangleMesh& mesh)
    : isReorderEnabled(true)
{
    build(soup, mesh);
}

void TriangleMeshBuilder::build(const TriangleSoup& soup, TriangleMesh& mesh)
{
    Log() << "Building mesh: " << soup.triangles.getSize() << " triangles, "
        << soup.triangles.getSize() * 3 << " vertices";

    constructMeshTopology(soup.triangles);

    if (isReorderEnabled)
        reorderFragments();

    storeMesh(mesh);
    cleanup();

    mesh.materialNames = soup.materialNames;
    mesh.bounds = bounds;
}

/*
void MeshBuilder::buildFast(const Array<FatTriangle>& triangles, TriangleMesh& mesh)
{
    Log() << "Building mesh (fast): " << triangles.size() << " triangles";

    constructMeshTopology(triangles);
    storeMesh(mesh);
    cleanup();
}
*/

void TriangleMeshBuilder::constructMeshTopology(const Array<FatTriangle>& triangles)
{
    Log() << "Constructing mesh topology";

    indexedTriangles.alloc(triangles.getSize());
    triangleFragments.alloc(triangles.getSize());
    bounds = empty;

    int vertexCount = 0;
    int totalVertexCount = triangles.getSize() * 3;

    // Scan for attribs
    hasNormals = false;
    hasTexcoords = false;
    hasMaterials = false;

    for (int iTri = 0; iTri < triangles.getSize(); ++iTri)
    {
        for (int iVert = 0; iVert < 3; ++iVert)
        {
            const FatVertex& vertex = triangles[iTri].v[iVert];

            if (vertex.normal != Vec3f(zero))
                hasNormals = true;

            if (vertex.tex != Vec2f(zero))
                hasTexcoords = true;
        }

        if (triangles[iTri].matId != 0)
            hasMaterials = true;
    }

    // Process tris
    FixedHashMap<FatVertex, int> vertexMap;
    vertexMap.alloc(totalVertexCount, min(totalVertexCount * 2, 64 * 1024 * 1024));

    for (int iTri = 0; iTri < triangles.getSize(); ++iTri)
    {
        Vec3f normal = triangles[iTri].getNormal();

        // Process the vertices
        for (int iVert = 0; iVert < 3; ++iVert)
        {
            FatVertex vertex = triangles[iTri].v[iVert];
            int existingVertexId;

            if (hasNormals && vertex.normal == Vec3f(zero))
            {
                // Fix normal
                vertex.normal = normal;
            }

            if (vertexMap.add(vertex, vertexCount, existingVertexId))
            {
                // New vertex
                vertices.pushBack(vertex);
                vertexFragments.pushBack(Fragment(vertex.pos, vertexCount));

                indexedTriangles[iTri].v[iVert] = vertexCount;

                ++vertexCount;
                bounds.grow(vertex.pos);
            }
            else
            {
                indexedTriangles[iTri].v[iVert] = existingVertexId;
            }
        }

        indexedTriangles[iTri].matId = triangles[iTri].matId;

        // Add the triangle
        triangleFragments[iTri] = Fragment(triangles[iTri].getCenter(), iTri);
    }

    Log() << "Unique vertices: " << vertexCount;
}

// Reorders the vertices and triangles to achieve a cache-oblivious layout
void TriangleMeshBuilder::reorderFragments()
{
    Log() << "Computing cache-oblivious mesh layout";
    reorderFragments(vertexFragments.begin(), vertexFragments.end(), triangleFragments.begin(), triangleFragments.end());
}

void TriangleMeshBuilder::reorderFragments(Fragment* vertexBegin, Fragment* vertexEnd, Fragment* triangleBegin, Fragment* triangleEnd)
{
    if (vertexBegin + 1 >= vertexEnd)
        return;

    // Compute the bounding box of the vertices
    Box3f box = empty;

    for (Fragment* i = vertexBegin; i < vertexEnd; ++i)
        box.grow(i->position);

    // Determine the splitting axis
    // We split along the largest dimension
    int splitAxis = selectMax(box.getSize());

    // We partition the vertices in two roughly equivalent size parts (median split)
    Fragment* vertexMiddle = vertexBegin + (vertexEnd - vertexBegin) / 2;
    //std::nth_element(vertexBegin, vertexMiddle, vertexEnd, [&](const Fragment& a, const Fragment& b) { return a.position[splitAxis] < b.position[splitAxis]; });

    // Quickselect
    {
        Fragment* first = vertexBegin;
        Fragment* last = vertexEnd;
        Fragment* nth = vertexMiddle;

        for (; ;)
        {
            Fragment* pivot = std::partition(first, last-1, [=](const Fragment& a) { return a.position[splitAxis] < (last-1)->position[splitAxis]; });
            std::iter_swap(last-1, pivot);

            if (pivot == nth)
                break;

            if (pivot > nth)
                last = pivot;
            else
                first = pivot+1;
        }
    }

    // We partition the triangles using the same plane, according to their centers
    float splitPosition = vertexMiddle->position[splitAxis];
    Fragment* triangleMiddle = triangleBegin;

    for (Fragment* i = triangleBegin; i < triangleEnd; ++i)
    {
        if (i->position[splitAxis] < splitPosition)
        {
            swap(*i, *triangleMiddle);
            ++triangleMiddle;
        }
    }

    // Reorder left part
    reorderFragments(vertexBegin, vertexMiddle, triangleBegin, triangleMiddle);

    // Reorder right part
    reorderFragments(vertexMiddle, vertexEnd, triangleMiddle, triangleEnd);
}

// Stores the vertices and triangles according to the reordered fragments
void TriangleMeshBuilder::storeMesh(TriangleMesh& mesh)
{
    Log() << "Assembling mesh";

    // Allocate memory
    int triangleCount = indexedTriangles.getSize();
    int vertexCount = vertices.getSize();
    mesh.alloc(triangleCount, vertexCount, hasNormals, hasTexcoords);

    // Store the vertices
    // We have to build a table that contains the IDs of the vertices after reordering
    // This maps the original IDs to the reordered IDs
    Array<int> newVertexIds;
    newVertexIds.alloc(vertexCount);

    for (int i = 0; i < vertexCount; ++i)
    {
        int oldVertexId = vertexFragments[i].id;
        const FatVertex& vertex = vertices[oldVertexId];

        newVertexIds[oldVertexId] = i;

        mesh.setPosition(i, vertex.pos);
        if (hasNormals) mesh.setNormal(i, vertex.normal);
        if (hasTexcoords) mesh.setTexcoord(i, vertex.tex);
    }

    // Store the triangles
    for (int iTri = 0; iTri < triangleCount; ++iTri)
    {
        const FatIndexedTriangle& oldTriangle = indexedTriangles[triangleFragments[iTri].id];

        // Get the new vertex IDs
        for (int iVert = 0; iVert < 3; ++iVert)
        {
            int oldVertexId = oldTriangle.v[iVert];
            int newVertexId = newVertexIds[oldVertexId];
            mesh.indices[iTri][iVert] = newVertexId;
        }

        mesh.materialIds[iTri] = hasMaterials ? oldTriangle.matId : 0;
    }
}

void TriangleMeshBuilder::cleanup()
{
    vertices.free();
    indexedTriangles.free();
    vertexFragments.free();
    triangleFragments.free();
}

} // namespace prt
