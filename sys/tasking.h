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

#include <omp.h>
#include "atomic.h"
#include "math/vec2.h"

namespace prt {

class Tasking
{
public:
    template <class Kernel>
    static void run(const Vec2i& gridSize, Kernel&& kernel)
    {
        AlignedAtomic<int> nextTaskId = 0;
        int taskCount = gridSize.x * gridSize.y;

        #pragma omp parallel
        {
            int threadId = omp_get_thread_num();

            for (; ;)
            {
                int taskId = nextTaskId++;
                if (taskId >= taskCount) break;

                Vec2i taskId2(taskId % gridSize.x, taskId / gridSize.x);
                kernel(taskId2, threadId);
            }
        }
    }

    static int getThreadCount()
    {
        return omp_get_max_threads();
    }
};

} // namespace prt
