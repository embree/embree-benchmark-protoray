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

#include "camera/pinhole_camera.h"
#include "camera/thin_lens_camera.h"
#include "renderer_factory_single.h"
#include "renderer_factory_packet.h"
#include "renderer_factory_stream.h"
#include "renderer_factory_stream_aos.h"
#include "device_cpu_impl.h"

namespace prt {

DeviceCpuImpl::DeviceCpuImpl()
    : rng(generateRandomSeed())
{
}

DeviceCpuImpl* DeviceCpuImpl::create()
{
    return new DeviceCpuImpl;
}

void DeviceCpuImpl::destroy(DeviceCpuImpl* impl)
{
    delete impl;
}

void DeviceCpuImpl::initScene(DeviceCpuImpl* impl, const std::string& path, const Props& props)
{
    impl->scene = makeRef<Scene>(path, props);
}

Box3f DeviceCpuImpl::getSceneBounds(DeviceCpuImpl* impl)
{
    return impl->scene->getBounds();
}

Props DeviceCpuImpl::initRenderer(DeviceCpuImpl* impl, const Props& props, const Props& stats)
{
    Props newStats = stats;
    std::string type = props.get("type");

    if (type.find("StreamAos") != std::string::npos)
        impl->renderer = RendererFactoryStreamAos::make(type, impl->scene, props, newStats);
    else if (type.find("Stream") != std::string::npos || type.find("Ms") != std::string::npos)
        impl->renderer = RendererFactoryStream::make(type, impl->scene, props, newStats);
    else if (type.find("Packet") != std::string::npos)
        impl->renderer = RendererFactoryPacket::make(type, impl->scene, props, newStats);
    else
        impl->renderer = RendererFactorySingle::make(type, impl->scene, props, newStats);

    return newStats;
}

Props DeviceCpuImpl::render(DeviceCpuImpl* impl, const Props& stats)
{
    Props newStats = stats;
    impl->renderer->render(impl->camera.get(), impl->frameBuffer.get(), newStats);
    return newStats;
}

Props DeviceCpuImpl::queryRay(DeviceCpuImpl* impl, const Ray& ray)
{
    return impl->renderer->queryRay(ray);
}

Props DeviceCpuImpl::queryPixel(DeviceCpuImpl* impl, int x, int y)
{
    return impl->renderer->queryPixel(impl->camera.get(), x, y);
}

void DeviceCpuImpl::initCamera(DeviceCpuImpl* impl, const Props& props)
{
    std::string type = props.get<std::string>("type");
    float lensRadius = props.get("lensRadius", 0.0f);

    if (type == "pinhole" || lensRadius == 0.0f)
        impl->camera = makeRef<PinholeCamera>(props);
    else if (type == "thinlens")
        impl->camera = makeRef<ThinLensCamera>(props);
    else
        throw std::invalid_argument("invalid camera type");
}

void DeviceCpuImpl::initFrame(DeviceCpuImpl* impl, const Vec2i& size, const Props& props)
{
    int accumFlags = 0;
    impl->frameBuffer = makeRef<FrameBuffer>(size, accumFlags);
}

void DeviceCpuImpl::initToneMapper(DeviceCpuImpl* impl, const Props& props)
{
    if (!impl->frameBuffer) return;

    ref<ToneMapper> toneMapper;
    std::string type = props.get<std::string>("type");

    if (type != "none")
        throw std::invalid_argument("invalid tone mapper type");

    impl->frameBuffer->setToneMapper(toneMapper);
}

void DeviceCpuImpl::clearFrame(DeviceCpuImpl* impl)
{
    impl->frameBuffer->clear();
}

void DeviceCpuImpl::blitFrame(DeviceCpuImpl* impl, Surface dest)
{
    impl->frameBuffer->blitLdr(dest);
}

} // namespace prt

