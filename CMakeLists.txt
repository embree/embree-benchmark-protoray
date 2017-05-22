## ======================================================================== ##
## Copyright 2015-2017 Intel Corporation                                    ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

project(protoray)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_COMPILER "icpc")

set(CMAKE_CXX_FLAGS "-std=c++11 -no-ansi-alias -qopenmp -static-intel -wd1875")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -no-inline-max-total-size -inline-factor=200")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g ${CMAKE_CXX_FLAGS_RELEASE}")

if(APPLE)
    set(CMAKE_CXX_COMPILER "/usr/local/bin/icpc")
    find_library(OPENGL_LIBRARY OpenGL)
endif()

set(HOST_ISA "CORE-AVX2" CACHE STRING "ISA")
set(HOST_CXX_FLAGS "-x${HOST_ISA}")

option(FORCE_SIMD8 "Force 8-wide SIMD" OFF)
if(FORCE_SIMD8)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFORCE_SIMD8")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

include_directories(${PROJECT_SOURCE_DIR})

option(EMBREE_SUPPORT "Embree support" ON)
if(EMBREE_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEMBREE_SUPPORT")
    set(EMBREE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../embree/include CACHE STRING "Embree include path")
    set(EMBREE_LIBRARY_DIR ${PROJECT_SOURCE_DIR}/../embree-build CACHE STRING "Embree library path")
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${EMBREE_INCLUDE_DIR}/../common/cmake")
    find_package(TBB REQUIRED)
    include_directories(${EMBREE_INCLUDE_DIR} ${TBB_INCLUDE_DIR})
    link_directories(${EMBREE_LIBRARY_DIR})
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

option(CUDA_SUPPORT "CUDA support" OFF)
if(CUDA_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_SUPPORT")
    find_package(CUDA REQUIRED)
    set(CUDA_HOST_COMPILER gcc)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -DNDEBUG -use_fast_math -Xptxas -v -keep -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62")
    include_directories(${CUDA_INCLUDE_DIRS})

    set(OPTIX_INSTALL_DIR /opt/optix CACHE STRING "OptiX install path")
    include_directories(${OPTIX_INSTALL_DIR}/include)
    link_directories(${OPTIX_INSTALL_DIR}/lib64)
endif()

if(APPLE)
    include_directories(/opt/local/include)
    link_directories(/opt/local/lib)
endif()

option(PROFILE_MODE "Performance counters" OFF)
if(PROFILE_MODE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPROFILE_MODE")
endif()

# Other switches
#set(HOST_CXX_FLAGS "${HOST_CXX_FLAGS} -S -masm=intel") # generate host asm

set(SRC_COMMON
    camera/camera.h
    camera/pinhole_camera.h
    camera/thin_lens_camera.h
    sys/array.h
    sys/atomic.h
    sys/blob.h
    sys/common.h
    sys/condition.h
    sys/constants.h
    sys/file.h
    sys/filesystem.cpp
    sys/filesystem.h
    sys/fixed_hash_map.h
    sys/lock_guard.h
    sys/logging.cpp
    sys/logging.h
    sys/mapped_file.cpp
    sys/mapped_file.h
    sys/memory.cpp
    sys/memory.h
    sys/mutex.h
    sys/option.cpp
    sys/option.h
    sys/props.h
    sys/raw_array.h
    sys/ref.h
    sys/static_array.h
    sys/stream.h
    sys/string.cpp
    sys/string.h
    sys/sysinfo.cpp
    sys/sysinfo.h
    sys/tasking.h
    sys/text_reader.cpp
    sys/text_reader.h
    sys/tick_counter.h
    sys/timer.h
    sys/value.h
    render/device.h
    render/device_cpu_impl.cpp
    render/device_cpu_impl.h
    render/integrator.h
    render/primary_integrator_single.h
    render/primary_integrator_packet.h
    render/ao_integrator_single.h
    render/ao_integrator_packet.h
    render/ao_hit_integrator_single.h
    render/po_integrator_single.h
    render/po_integrator_packet.h
    render/diffuse_integrator_single.h
    render/diffuse_integrator_packet.h
    render/debug_integrator_single.h
    render/renderer.h
    render/renderer_single.h
    render/renderer_packet.h
    render/renderer_stream.h
    render/primary_renderer_stream.h
    render/diffuse_renderer_stream.h
    render/intersector_factory_single.cpp
    render/intersector_factory_single.h
    render/intersector_factory_packet.cpp
    render/intersector_factory_packet.h
    render/intersector_factory_stream.h
    render/renderer_factory.cpp
    render/renderer_factory.h
    render/renderer_factory_single.cpp
    render/renderer_factory_single.h
    render/renderer_factory_packet.cpp
    render/renderer_factory_packet.h
    render/renderer_factory_stream.cpp
    render/renderer_factory_stream.h
    render/scene.cpp
    render/scene.h
    geometry/shape.h
    geometry/triangle.cpp
    geometry/triangle.h
    geometry/triangle_mesh.cpp
    geometry/triangle_mesh.h
    geometry/triangle_soup.cpp
    geometry/triangle_soup.h
    geometry/quad.h
    geometry/vertex.h
    image/color.h
    image/frame_buffer.cpp
    image/frame_buffer.h
    image/image.cpp
    image/image.h
    image/image_io.h
    image/morton.cpp
    image/morton.h
    image/pixel.cpp
    image/pixel.h
    image/tone_mapper.h
    image/linear_tone_mapper.h
    image/reinhard_tone_mapper.h
    math/basis3.h
    math/box1.h
    math/box2.h
    math/box3.h
    math/fast_box3.h
    math/fast_random.h
    math/fast_random_simd.h
    math/math_common.h
    math/math.h
    math/random.h
    math/simd_common.cpp
    math/simd_common.h
    math/simd.h
    math/vec2.h
    math/vec3.h
    math/vec4.h
    core/intersector.h
    core/intersector_single.h
    core/intersector_packet.h
    core/intersector_stream.h
    core/post_intersector.h
    core/ray.h
    core/ray_simd.h
    core/ray_stream.h
    core/ray_stats.h
    core/shading_context.h
    sampling/shape_sampler.h
    sampling/random_sampler.h
    sampling/random_sampler_simd.h
)

set(SRC_HOST
    ${SRC_COMMON}
    sys/file.cpp
    render/device_cpu.cpp
    render/device_cpu.h
    geometry/triangle_mesh_builder.cpp
    geometry/triangle_mesh_builder.h
    image/image_io.cpp
    main/build/build.cpp
    main/build/obj_loader.cpp
    main/build/obj_loader.h
    main/main.h
    main/main.cpp
    main/render/input.h
    main/render/render.cpp
    main/render/render_window.cpp
    main/render/render_window.h
    main/render/stats_recorder.cpp
    main/render/stats_recorder.h
    main/render/view.cpp
    main/render/view.h
    main/render/window.cpp
    main/render/window.h
    math/math_avx.h
    math/simd/vfloat16_avx512.h
    math/simd/vfloat16_avx.h
    math/simd/vfloat4_avx.h
    math/simd/vfloat8_avx.h
    math/simd/vint16_avx512.h
    math/simd/vint16_avx.h
    math/simd/vint4_avx.h
    math/simd/vint8_avx.h
    math/simd/vint8_avx2.h
    math/simd/vbool16_avx512.h
    math/simd/vbool16_avx.h
    math/simd/vbool4_avx.h
    math/simd/vbool8_avx.h
)

if(EMBREE_SUPPORT)
    set(SRC_HOST
        ${SRC_HOST}
        accel/embree/embree_intersector.cpp
        accel/embree/embree_intersector.h
        accel/embree/embree_intersector_single.h
        accel/embree/embree_intersector_packet.h
        accel/embree/embree_intersector_stream.h
    )
endif()

if(CUDA_SUPPORT)
    set(SRC_HOST
        ${SRC_HOST}
        accel/optix/optix_intersector_stream_cuda.h
        accel/optix/optix_intersector_stream_cuda.cpp
        camera/camera_cuda.h
        camera/pinhole_camera_cuda.h
        camera/pinhole_camera_cuda.cuh
        camera/thin_lens_camera_cuda.h
        camera/thin_lens_camera_cuda.cuh
        core/ray_cuda.cuh
        core/shading_context_cuda.cuh
        core/intersector_cuda.cuh
        core/intersector_stream_cuda.h
        math/math.cuh
        image/accum_buffer_cuda.cu
        image/accum_buffer_cuda.h
        image/frame_buffer_cuda.cpp
        image/frame_buffer_cuda.h
        render/device_cuda.cpp
        render/device_cuda.h
        render/renderer_cuda.h
        render/primary_renderer_cuda.cu
        render/primary_renderer_cuda.h
        render/diffuse_renderer_cuda.cu
        render/diffuse_renderer_cuda.h
        sampling/random_sampler.cuh
        sampling/shape_sampler.cuh
        sys/common.cuh
        sys/cuda.h
        sys/buffer_cuda.h
        sys/timer_cuda.h
    )
endif()

if(CUDA_SUPPORT)
    #target_link_libraries(protoray cudart_static)
    cuda_add_executable(protoray ${SRC_HOST})
    target_link_libraries(protoray optix_prime)
else()
    add_executable(protoray ${SRC_HOST})
endif()

set_target_properties(protoray PROPERTIES COMPILE_FLAGS ${HOST_CXX_FLAGS})
if(NOT APPLE)
    target_link_libraries(protoray rt)
endif()

if(EMBREE_SUPPORT)
    target_link_libraries(protoray embree ${TBB_LIBRARIES})
endif()
