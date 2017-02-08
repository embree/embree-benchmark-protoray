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

#include <iostream>
#include "sys/string.h"
#include "sys/logging.h"
#include "sys/memory.h"
#include "sys/sysinfo.h"
#include "sys/filesystem.h"
#include "sys/option.h"

#include "render/device_cpu.h"
#ifdef CUDA_SUPPORT
#include "render/device_cuda.h"
#endif

#include "main/main.h"
#include "render_window.h"

namespace prt {

namespace {

std::string sceneFilename;
std::string deviceId = "cpu";
std::string rendererId = "diffuse";
DisplayMode displayMode = displayModeOffscreen;
Vec2i imageSize(1024, 576);
int viewId = 0;
bool isBenchmarkMode = false;
std::string resultPrefix;
Props props;

bool applyOptions(const Array<Option>& opts)
{
    for (const Option& opt : opts)
    {
        if (opt.name.empty())
        {
            sceneFilename = opt.value;
        }
        else if (opt.name == "c" || opt.name == "config")
        {
            Array<Option> opts2;
            parseOptions(opt.value, opts2);
            if (!applyOptions(opts2))
                return false;
        }
        else if (opt.name == "dev" || opt.name == "device" || opt.name == "e" || opt.name == "engine")
        {
            deviceId = opt.value;
        }
        else if (opt.name == "r" || opt.name == "renderer")
        {
            rendererId = opt.value;
        }
        else if (opt.name == "a" || opt.name == "accel")
        {
            props.set("accel", opt.value);
        }
        else if (opt.name == "i" || opt.name == "isect")
        {
            props.set("isect", opt.value);
        }
        else if (opt.name == "s" || opt.name == "size")
        {
            imageSize = opt.value.get<Vec2i>();
        }
        else if (opt.name == "dw")
        {
            displayMode = displayModeWindow;
        }
        else if (opt.name == "do")
        {
            displayMode = displayModeOffscreen;
        }
        else if (opt.name == "df")
        {
            displayMode = displayModeFullscreen;
        }
        else if (opt.name == "benchmark" || opt.name == "bench")
        {
            isBenchmarkMode = true;
            resultPrefix = opt.value;
            if (resultPrefix.empty())
                resultPrefix = "benchmark";
            props.set(opt.name, opt.value);
        }
        else if (opt.name == "view")
        {
            viewId = opt.value.get<int>();

            if (viewId < 0 || viewId >= ViewSet::size)
            {
                LogError() << "Invalid view";
                return false;
            }
        }
        else if (opt.name == "temp")
        {
            setTempPath(opt.value);
        }
        else
        {
            props.set(opt.name, opt.value);
        }
    }

    return true;
}

} // namespace

int mainRender(int argc, char* argv[])
{
    std::cout << "ProtoRay Render" << std::endl;
	std::cout << std::endl;

    // Parse the options
    if (argc < 2)
    {
        std::cout << "Usage: protoray render [options]" << std::endl;
        return 0;
    }

    Array<Option> opts;
    parseOptions(argc, argv, opts);
    if (!applyOptions(opts))
        return 1;

    if (sceneFilename.empty())
	{
        LogError() << "Scene not specified";
		return 1;
	}

    std::string sceneBase = getFilenameBase(sceneFilename);
    std::string sceneName = sceneBase.substr(sceneBase.find_last_of("/\\") + 1);

    // Init logging to file
    if (isBenchmarkMode)
        setLogFile(resultPrefix + ".log");
    else
        setLogFile("render.log");

    #ifdef DEBUG
        LogWarn() << "Debug build";
    #endif

	// Log command line
    std::string commandLine = std::string(argv[0]) + " render ";
    for (int i = 1; i < argc; ++i)
		 commandLine += std::string(argv[i]) + " ";

	Log() << "Command line: " << commandLine;

    // Check the display settings
    if (imageSize.x <= 0 || imageSize.x % 8 != 0 ||
        imageSize.y <= 0 || imageSize.y % 8 != 0)
	{
        LogError() << "Resolution must be multiple of 8";
		return 1;
	}

	// CPU info
	CpuInfo cpuInfo;
	getCpuInfo(cpuInfo);
	Log() << "CPU: " << cpuInfo.brand;
    Log() << "CPU threads: " << getCpuCount();
    Log() << "CPU SIMD width: " << simdSize;

    // Initialize device
    const char* binPath = argv[0];
    ref<Device> device;

    if (deviceId == "cpu")
    {
        device = makeRef<DeviceCpu>();
    }
#ifdef CUDA_SUPPORT
    else if (deviceId == "cuda")
    {
        device = makeRef<DeviceCuda>();
    }
#endif
    else
    {
        LogError() << "Invalid device: " << deviceId;
        return 1;
    }

    std::string deviceInfo = device->getInfo();
    if (deviceInfo != cpuInfo.brand)
        Log() << "Device: " << deviceInfo;

    device->initScene(sceneFilename, props);

    Props rendererProps = props;
    rendererProps.set("type", rendererId);
    rendererProps.set("imageSize", imageSize);
    Props buildStats;
    device->initRenderer(rendererProps, buildStats);

    device->initFrame(imageSize);

	// Create and start the UI
    props.set("viewFile", sceneBase + ".view");
    props.set("view", viewId);

	if (isBenchmarkMode)
        props.set("benchmark", resultPrefix);

    RenderWindow window(imageSize.x, imageSize.y, displayMode, device, props, buildStats);
    window.setTitle("ProtoRay");
	window.run();

	return 0;
}

} // namespace prt
