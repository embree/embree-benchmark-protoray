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

class MemoryCuda
{
private:
    void* data;
    size_t size;

public:
    FORCEINLINE MemoryCuda() : data(0), size(0) {}

    MemoryCuda(size_t size)
    {
      if (cudaMalloc(&data, size) != cudaSuccess)
          throw std::runtime_error("cudaMalloc failed");

        this->size = size;
    }

    MemoryCuda(const MemoryCuda& other)
    {
        if (other.data)
        {
            if (cudaMalloc(&data, other.size) != cudaSuccess)
                throw std::runtime_error("cudaMalloc failed");
            size = other.size;
            cudaMemcpy(data, other.data, size, cudaMemcpyDeviceToDevice);
        }
        else
        {
            data = 0;
            size = 0;
        }
    }

    MemoryCuda(const Memory& other)
    {
        if (other.getData())
        {
            if (cudaMalloc(&data, other.getSize()) != cudaSuccess)
                throw std::runtime_error("cudaMalloc failed");
            size = other.getSize();
            cudaMemcpy(data, other.getData(), size, cudaMemcpyHostToDevice);
        }
        else
        {
            data = 0;
            size = 0;
        }
    }

    FORCEINLINE ~MemoryCuda()
    {
        if (data) free();
    }

    MemoryCuda& operator =(const MemoryCuda& other)
    {
        if (this != &other)
        {
            if (data) free();

            if (other.data)
            {
                if (cudaMalloc(&data, other.size) != cudaSuccess)
                    throw std::runtime_error("cudaMalloc failed");
                size = other.size;
                cudaMemcpy(data, other.data, size, cudaMemcpyDeviceToDevice);
            }
        }

        return *this;
    }

    MemoryCuda& operator =(const Memory& other)
    {
        if (data) free();

        if (other.getData())
        {
            if (cudaMalloc(&data, other.getSize()) != cudaSuccess)
                throw std::runtime_error("cudaMalloc failed");
            size = other.getSize();
            cudaMemcpy(data, other.getData(), size, cudaMemcpyHostToDevice);
        }

        return *this;
    }

    void alloc(size_t size)
    {
        if (data) free();

        if (cudaMalloc(&data, size) != cudaSuccess)
            throw std::runtime_error("cudaMalloc failed");

        this->size = size;
    }

    void free()
    {
        if (cudaFree(data) != cudaSuccess)
            throw std::runtime_error("cudaFree failed");

        data = 0;
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
};

} // namespace prt
