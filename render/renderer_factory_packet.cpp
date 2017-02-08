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

#include "sampling/random_sampler_simd.h"
#include "intersector_factory_packet.h"
#include "renderer_packet.h"
#include "primary_integrator_packet.h"
#include "po_integrator_packet.h"
#include "ao_integrator_packet.h"
#include "diffuse_integrator_packet.h"
#include "renderer_factory_packet.h"

namespace prt {

ref<Renderer> RendererFactoryPacket::make(const std::string& type, const ref<const Scene>& scene, const Props& props, Props& stats)
{
    std::string samplerType = props.get("sampler", "random");

    if (samplerType == "random")
        return makeWithSampler<RandomSamplerSimd>(type, scene, props, stats);

    throw std::invalid_argument("invalid sampler type");
}

template <class Sampler>
ref<Renderer> RendererFactoryPacket::makeWithSampler(const std::string& type, const ref<const Scene>& scene, const Props& props, Props& stats)
{
    ref<IntersectorPacket> intersector = IntersectorFactoryPacket::make(scene, props, stats);

    if (type == "primaryPacket")
        return makeRef<RendererPacket<PrimaryIntegratorPacket<ShadingContextSimd, Sampler>, Sampler>>(scene, intersector, props);
    if (type == "primaryPacketFast")
        return makeRef<RendererPacket<PrimaryIntegratorPacket<SimpleShadingContextSimd, Sampler>, Sampler, false>>(scene, intersector, props);
    if (type == "poPacket")
        return makeRef<RendererPacket<PoIntegratorPacket<Sampler>, Sampler>>(scene, intersector, props);
    if (type == "aoPacket")
        return makeRef<RendererPacket<AoIntegratorPacket<Sampler>, Sampler>>(scene, intersector, props);
    if (type == "diffusePacket")
        return makeRef<RendererPacket<DiffuseIntegratorPacket<ShadingContextSimd, Sampler>, Sampler>>(scene, intersector, props);
    if (type == "diffusePacketFast")
        return makeRef<RendererPacket<DiffuseIntegratorPacket<SimpleShadingContextSimd, Sampler>, Sampler, false>>(scene, intersector, props);

    throw std::invalid_argument("invalid renderer type");
}

} // namespace prt
