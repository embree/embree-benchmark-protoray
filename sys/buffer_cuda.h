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

#include <cuda_runtime.h>
#include "memory.h"

namespace prt {

class BufferCuda : Uncopyable
{
private:
    void* data;       // device
    void* shadowData; // host
    size_t size;
    Access mapAccess;

public:
    FORCEINLINE BufferCuda() : data(0), shadowData(0), size(0) {}

    FORCEINLINE ~BufferCuda()
    {
        if (data) free();
    }

    void alloc(size_t size)
    {
        if (data) free();

        if (cudaMalloc(&data, size) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");

        if (cudaMallocHost(&shadowData, size) != cudaSuccess)
            throw std::runtime_error("cudaMallocHost failed");

        this->size = size;
    }

    void free()
    {
        if (cudaFree(data) != cudaSuccess)
            throw std::runtime_error("cudaFree failed");

        data = 0;

        if (cudaFreeHost(shadowData) != cudaSuccess)
            throw std::runtime_error("cudaFreeHost failed");

        shadowData = 0;
        size = 0;
    }

    FORCEINLINE void* getData()
    {
        return data;
    }

    FORCEINLINE const void* getData() const
    {
        return data;
    }

    FORCEINLINE size_t getSize() const
    {
        return size;
    }

    void* map(Access access)
    {
        if (access != accessReadWriteDiscard)
            cudaMemcpy(shadowData, data, size, cudaMemcpyDeviceToHost);

        mapAccess = access;
        return shadowData;
    }

    void unmap()
    {
        if (mapAccess != accessRead)
            cudaMemcpy(data, shadowData, size, cudaMemcpyHostToDevice);
    }
};

} // namespace prt
