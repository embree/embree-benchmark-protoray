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

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#include <sys/resource.h>
#ifdef __INTEL_COMPILER
#include <ia32intrin.h>
#endif
#endif

#include "logging.h"
#include "sysinfo.h"

//#define SINGLE_THREADED

namespace prt {

const int cpuCount = getCpuCount();

#if defined(__AVX512F__)
    const std::string cpuIsa = "avx512";
#elif defined(__AVX2__)
    const std::string cpuIsa = "avx2";
#elif defined(__AVX__)
    const std::string cpuIsa = "avx";
#elif defined(__SSE__)
    const std::string cpuIsa = "sse";
#elif defined(__MIC__)
    const std::string cpuIsa = "knc";
#else
    const std::string cpuIsa = "base";
#endif

int getCpuCount()
{
#if defined(SINGLE_THREADED)
	return 1;
#elif defined(_WIN32)
	SYSTEM_INFO systemInfo;
	GetSystemInfo(&systemInfo);
	return static_cast<int>(systemInfo.dwNumberOfProcessors);
#else
	return static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
#endif
}

void forceCpuCount(int n)
{
    const_cast<int&>(cpuCount) = n;
}

void getCpuInfo(CpuInfo& info)
{
#if defined(__MIC__) || !defined(__INTEL_COMPILER)
    info.brand = "Unknown";
#else
	char cpuBrandString[0x40];
    int cpuInfo[4] = {-1};
	uint32_t nExIds, i;
   
    __cpuid(cpuInfo, 0x80000000);
    nExIds = cpuInfo[0];
    memset(cpuBrandString, 0, sizeof(cpuBrandString));

    for (i = 0x80000000; i <= nExIds; ++i)
    {
        __cpuid(cpuInfo, i);

        if (i == 0x80000002)
            memcpy(cpuBrandString, cpuInfo, sizeof(cpuInfo));
        else if (i == 0x80000003)
            memcpy(cpuBrandString + 16, cpuInfo, sizeof(cpuInfo));
        else if (i == 0x80000004)
            memcpy(cpuBrandString + 32, cpuInfo, sizeof(cpuInfo));
    }

	if (nExIds >= 0x80000004)
	{
		const char* s = cpuBrandString;
        while (isspace(*s)) s++;
		info.brand = s;
	}
	else
	{
		info.brand = "Unknown";
	}
#endif
}

} // namespace prt
