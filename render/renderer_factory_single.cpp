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
#include "intersector_factory_single.h"
#include "renderer_single.h"
#include "primary_integrator_single.h"
#include "po_integrator_single.h"
#include "ao_integrator_single.h"
#include "ao_hit_integrator_single.h"
#include "diffuse_integrator_single.h"
#include "diffuse2_integrator_single.h"
#include "debug_integrator_single.h"
#include "renderer_factory_single.h"

namespace prt {

ref<Renderer> RendererFactorySingle::make(const std::string& type, const ref<const Scene>& scene, const Props& props, Props& stats)
{
    std::string samplerType = getSamplerType(type, props);

    if (samplerType == "random")
        return makeWithSampler<RandomSampler>(type, scene, props, stats);

    throw std::invalid_argument("invalid sampler type");
}

template <class Sampler>
ref<Renderer> RendererFactorySingle::makeWithSampler(const std::string& type, const ref<const Scene>& scene, const Props& props, Props& stats)
{
    ref<IntersectorSingle> intersector = IntersectorFactorySingle::make(scene, props, stats);

    if (type == "primary")
        return makeRef<RendererSingle<PrimaryIntegratorSingle<ShadingContext, Sampler>, Sampler>>(scene, intersector, props);
    if (type == "primaryFast")
        return makeRef<RendererSingle<PrimaryIntegratorSingle<SimpleShadingContext, Sampler>, Sampler, false>>(scene, intersector, props);
    if (type == "po")
        return makeRef<RendererSingle<PoIntegratorSingle<Sampler>, Sampler>>(scene, intersector, props);
    if (type == "ao")
        return makeRef<RendererSingle<AoIntegratorSingle<Sampler>, Sampler>>(scene, intersector, props);
    if (type == "aoHit")
        return makeRef<RendererSingle<AoHitIntegratorSingle<Sampler>, Sampler>>(scene, intersector, props);
    if (type == "diffuse")
        return makeRef<RendererSingle<DiffuseIntegratorSingle<ShadingContext, Sampler>, Sampler>>(scene, intersector, props);
    if (type == "diffuseFast")
        return makeRef<RendererSingle<DiffuseIntegratorSingle<SimpleShadingContext, Sampler>, Sampler, false>>(scene, intersector, props);
    if (type == "diffuse2")
        return makeRef<RendererSingle<Diffuse2IntegratorSingle<ShadingContext, Sampler>, Sampler>>(scene, intersector, props);
    if (type == "diffuse2Fast")
        return makeRef<RendererSingle<Diffuse2IntegratorSingle<SimpleShadingContext, Sampler>, Sampler, false>>(scene, intersector, props);
    if (type == "debug")
        return makeRef<RendererSingle<DebugIntegratorSingle<Sampler>, Sampler>>(scene, intersector, props);

    throw std::invalid_argument("invalid renderer type");
}

} // namespace prt
