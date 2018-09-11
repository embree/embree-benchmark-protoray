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

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <pthread.h>
#endif

#include "common.h"

namespace prt {

class Mutex : Uncopyable
{
private:
#ifdef _WIN32
    CRITICAL_SECTION handle;
#else
    pthread_mutex_t handle;
#endif

public:
	friend class Condition;

	Mutex()
	{
#ifdef _WIN32
        InitializeCriticalSection(&handle);
#else
        int result = pthread_mutex_init(&handle, NULL);
		assert(result == 0 && "Could not create mutex.");
#endif
	}

	~Mutex()
	{
#ifdef _WIN32
        DeleteCriticalSection(&handle);
#else
        pthread_mutex_destroy(&handle);
#endif
	}

	void lock()
	{
#ifdef _WIN32
        EnterCriticalSection(&handle);
#else
        int result = pthread_mutex_lock(&handle);
		assert(result == 0 && "Could not lock mutex.");
#endif
	}

	bool tryLock()
	{
#ifdef _WIN32
        return TryEnterCriticalSection(&handle) != 0;
#else
        return pthread_mutex_trylock(&handle) == 0;
#endif
	}

	void unlock()
	{
#ifdef _WIN32
        LeaveCriticalSection(&handle);
#else
        int result = pthread_mutex_unlock(&handle);
		assert(result == 0 && "Could not unlock mutex.");
#endif
	}
};

} // namespace prt
