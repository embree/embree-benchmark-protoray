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

#pragma once

#include "mutex.h"

namespace prt {

class Condition : Uncopyable
{
private:
#ifdef _WIN32
    CONDITION_VARIABLE handle;
#else
    pthread_cond_t handle;
#endif

public:
    Condition()
	{
#ifdef _WIN32
        InitializeConditionVariable(&handle);
#else
        int result = pthread_cond_init(&handle, NULL);
		assert(result == 0 && "Could not create condition variable.");
#endif
	}

    ~Condition()
	{
#ifndef _WIN32
        pthread_cond_destroy(&handle);
#endif
	}

	void notifyOne()
	{
#ifdef _WIN32
        WakeConditionVariable(&handle);
#else
        int result = pthread_cond_signal(&handle);
		assert(result == 0 && "Could not notify a thread waiting on the condition variable.");
#endif
	}

	void notifyAll()
	{
#ifdef _WIN32
        WakeAllConditionVariable(&handle);
#else
        int result = pthread_cond_broadcast(&handle);
		assert(result == 0 && "Could not notify all threads waiting on the condition variable.");
#endif
	}

	void wait(Mutex& mutex)
	{
#ifdef _WIN32
        BOOL result = SleepConditionVariableCS(&handle, &mutex.handle, INFINITE);
		assert(result != 0 && "Could not sleep on the condition variable.");
#else
        int result = pthread_cond_wait(&handle, &mutex.handle);
		assert(result == 0 && "Could not sleep on the condition variable.");
#endif
	}
};

} // namespace prt
