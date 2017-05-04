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

#pragma once

#include "common.h"

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__INTEL_COMPILER)
#include <ia32intrin.h>
#endif

namespace prt {

class TickCounter
{
private:
    uint64_t startCount;

public:
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
	TickCounter()
	{
		reset();
	}

	void reset()
	{
		int cpuInfo[4];
		__cpuid(cpuInfo, 0);

        startCount = __rdtsc();
	}

	uint64_t query() const
	{
        //unsigned int aux;
        //uint64_t currentCount = __rdtscp(&aux);
        uint64_t currentCount = __rdtsc();

		int cpuInfo[4];
		__cpuid(cpuInfo, 0);

        return currentCount - startCount;
	}

    static uint64_t now()
    {
        //unsigned int aux;
        //return __rdtscp(&aux);
        return __rdtsc();
    }
#else
    static uint64_t now()
    {
        uint64_t x;
         __asm__ volatile ("rdtsc" : "=A" (x));
         return x;
    }
#endif
};

} // namespace prt
