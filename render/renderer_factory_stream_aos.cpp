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

#include "sampling/random_sampler.h"
#include "intersector_factory_stream_aos.h"
#include "primary_renderer_stream_aos.h"
#include "diffuse_renderer_stream_aos.h"
#include "diffuse2_renderer_stream_aos.h"
#include "ao_renderer_stream_aos.h"
#include "ao_common_renderer_stream_aos.h"
#include "renderer_factory_stream_aos.h"

namespace prt {

ref<Renderer> RendererFactoryStreamAos::make(const std::string& type, const ref<const Scene>& scene, const Props& props, Props& stats)
{
    std::string samplerType = getSamplerType(type, props);

    if (samplerType == "random")
        return makeWithSampler<RandomSampler>(type, scene, props, stats);

    throw std::invalid_argument("invalid sampler type");
}

template <class Sampler>
ref<Renderer> RendererFactoryStreamAos::makeWithSampler(const std::string& type, const ref<const Scene>& scene, const Props& props, Props& stats)
{
    int streamSize = props.get("streamSize", defaultRayStreamSize);

#if 1
    static_assert(defaultRayStreamSize == 2048 || defaultRayStreamSize == 4096, "invalid defaultRayStreamSize");
    if (streamSize == 256)
        return makeWithSamplerStream<Sampler, 256, 16>(type, scene, props, stats);
    if (streamSize == defaultRayStreamSize)
        return makeWithSamplerStream<Sampler, defaultRayStreamSize, 64>(type, scene, props, stats);
#else
#if SIMD_REG_SIZE == 32
    if (streamSize == 8)
        return makeWithSamplerStream<Sampler, 8, 8>(type, scene, props, stats);
#endif
    if (streamSize == 16)
        return makeWithSamplerStream<Sampler, 16, 8>(type, scene, props, stats);
    if (streamSize == 32)
        return makeWithSamplerStream<Sampler, 32, 8>(type, scene, props, stats);
    if (streamSize == 64)
        return makeWithSamplerStream<Sampler, 64, 8>(type, scene, props, stats);
    if (streamSize == 128)
        return makeWithSamplerStream<Sampler, 128, 16>(type, scene, props, stats);
    if (streamSize == 256)
        return makeWithSamplerStream<Sampler, 256, 16>(type, scene, props, stats);
    if (streamSize == 512)
        return makeWithSamplerStream<Sampler, 512, 32>(type, scene, props, stats);
    if (streamSize == 1024)
        return makeWithSamplerStream<Sampler, 1024, 32>(type, scene, props, stats);
    if (streamSize == 2048)
        return makeWithSamplerStream<Sampler, 2048, 64>(type, scene, props, stats);
    if (streamSize == 4096)
        return makeWithSamplerStream<Sampler, 4096, 64>(type, scene, props, stats);
    if (streamSize == 8192)
        return makeWithSamplerStream<Sampler, 8192, 128>(type, scene, props, stats);
    if (streamSize == 16384)
        return makeWithSamplerStream<Sampler, 16384, 128>(type, scene, props, stats);
    if (streamSize == 32768)
        return makeWithSamplerStream<Sampler, 32768, 256>(type, scene, props, stats);
#endif

    throw std::invalid_argument("invalid stream size");
}

template <class Sampler, int streamSize, int streamTileX>
ref<Renderer> RendererFactoryStreamAos::makeWithSamplerStream(const std::string& type, const ref<const Scene>& scene, const Props& props, Props& stats)
{
    static const int streamTileY = (streamSize > 64) ? (streamSize / streamTileX) : (64 / streamTileX);

    // Stream AOS
    Log() << "AOS stream size: " << streamSize;

    ref<IntersectorStreamAos<streamSize>> intersector = IntersectorFactoryStreamAos::make<streamSize>(scene, props, stats);

    if (type == "primaryStreamAos")
        return makeRef<PrimaryRendererStreamAos<ShadingContext, Sampler, true, streamSize, streamTileX, streamTileY>>(scene, intersector, props);
    if (type == "primaryStreamAosFast")
        return makeRef<PrimaryRendererStreamAos<SimpleShadingContext, Sampler, false, streamSize, streamTileX, streamTileY>>(scene, intersector, props);
    if (type == "diffuseStreamAos")
        return makeRef<DiffuseRendererStreamAos<ShadingContext, Sampler, true, streamSize, streamTileX, streamTileY>>(scene, intersector, props);
    if (type == "diffuseStreamAosFast")
        return makeRef<DiffuseRendererStreamAos<SimpleShadingContext, Sampler, false, streamSize, streamTileX, streamTileY>>(scene, intersector, props);
    if (type == "diffuse2StreamAos")
        return makeRef<Diffuse2RendererStreamAos<ShadingContext, Sampler, true, streamSize, streamTileX, streamTileY>>(scene, intersector, props);
    if (type == "diffuse2StreamAosFast")
        return makeRef<Diffuse2RendererStreamAos<SimpleShadingContext, Sampler, false, streamSize, streamTileX, streamTileY>>(scene, intersector, props);
    if (type == "aoStreamAos")
        return makeRef<AoRendererStreamAos<ShadingContext, Sampler, true, streamSize, streamTileX, streamTileY>>(scene, intersector, props);
    if (type == "aoStreamAosFast")
        return makeRef<AoRendererStreamAos<SimpleShadingContext, Sampler, false, streamSize, streamTileX, streamTileY>>(scene, intersector, props);
    if (type == "aoCommonStreamAos")
        return makeRef<AoCommonRendererStreamAos<ShadingContext, Sampler, true, streamSize, streamTileX, streamTileY>>(scene, intersector, props);
    if (type == "aoCommonStreamAosFast")
        return makeRef<AoCommonRendererStreamAos<SimpleShadingContext, Sampler, false, streamSize, streamTileX, streamTileY>>(scene, intersector, props);

    throw std::invalid_argument("invalid renderer type");
}

} // namespace prt

