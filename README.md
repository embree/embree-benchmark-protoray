Compiling ProtoRay
==================

Requirements: Linux, CMake, ICC, GCC, CUDA, Embree, OptiX

By default, you have to put the Embree source directory 'embree' in the same directory as the ProtoRay source directory, and the Embree build directory must be called 'embree-build'. These paths can be changed in CMake.

You can set the target ISA with the HOST_ISA CMake option (e.g., CORE-AVX2 for HSW, MIC-AVX512 for KNL, CORE-AVX512 for SKX).


Running ProtoRay
================

Before rendering a scene in .obj format, you have to build a corresponding .mesh file:

    ./protoray build scene.obj


Render with single-ray diffuse path tracing:

    ./protoray render scene.mesh -no-mtl -r diffuse


Render with packet diffuse path tracing:

    ./protoray render scene.mesh -no-mtl -r diffusePacket -size 1920,1080


Render with stream diffuse path tracing:

    ./protoray render scene.mesh -no-mtl -r diffuseStream -streamSize 256 -size 1920,1080
