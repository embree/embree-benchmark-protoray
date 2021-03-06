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

#ifdef CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

#include <iostream>
#include "sys/string.h"
#include "sys/logging.h"
#include "sys/sysinfo.h"
#include "image/image_io.h"
#include "main.h"

namespace prt {

namespace
{
    std::string binPath;

    void printHeader()
    {
        std::cout << "ProtoRay" << std::endl;
        std::cout << std::endl;
    }
}

} // namespace prt

int main(int argc, char* argv[])
{
    using namespace prt;

	if (argc == 1)
	{
		printHeader();
        std::cout << "Commands: build, render" << std::endl;
		return 0;
	}

#if CUDA_SUPPORT
    // Workaround for occasional CUDA freeze
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
#endif

    initLogging();

    std::string appName = argv[1];
    binPath = argv[0];
	argv[1] = argv[0];

    if (appName == "build") return mainBuild(argc - 1, argv + 1);
    if (appName == "render") return mainRender(argc - 1, argv + 1);
	printHeader();
    std::cout << "Invalid command!";
	return 1;
}
