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

#include "sys/sysinfo.h"
#include "device_cpu.h"

namespace prt {

DeviceCpu::DeviceCpu()
{
    impl = DeviceCpuImpl::create();
}

DeviceCpu::~DeviceCpu()
{
    DeviceCpuImpl::destroy(impl);
}

std::string DeviceCpu::getInfo()
{
    CpuInfo cpuInfo;
    getCpuInfo(cpuInfo);
    return cpuInfo.brand;
}

void DeviceCpu::initRenderer(const Props& props, Props& stats)
{
    stats = DeviceCpuImpl::initRenderer(impl, props, stats);
}

void DeviceCpu::render(Props& stats)
{
    stats = DeviceCpuImpl::render(impl, stats);
}

Props DeviceCpu::queryRay(const Ray& ray)
{
    return DeviceCpuImpl::queryRay(impl, ray);
}

Props DeviceCpu::queryPixel(int x, int y)
{
    return DeviceCpuImpl::queryPixel(impl, x, y);
}

void DeviceCpu::initScene(const std::string& path, const Props& props)
{
    DeviceCpuImpl::initScene(impl, path, props);
}

Box3f DeviceCpu::getSceneBounds()
{
    return DeviceCpuImpl::getSceneBounds(impl);
}

void DeviceCpu::initCamera(const Props& props)
{
    DeviceCpuImpl::initCamera(impl, props);
}

void DeviceCpu::initFrame(const Vec2i& size, const Props& props)
{
    DeviceCpuImpl::initFrame(impl, size, props);
}

void DeviceCpu::initToneMapper(const Props& props)
{
    DeviceCpuImpl::initToneMapper(impl, props);
}

void DeviceCpu::clearFrame()
{
    DeviceCpuImpl::clearFrame(impl);
}

void DeviceCpu::blitFrame(Surface& dest)
{
    DeviceCpuImpl::blitFrame(impl, dest);
}

} // namespace prt


