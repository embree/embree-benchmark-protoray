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
// Windows
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

#include "common.h"

namespace prt {

class MappedFile : Uncopyable
{
private:
    void* data_;
    size_t size_;
    Access access;

#ifdef _WIN32
    // Windows
    HANDLE fileHandle;
    HANDLE mappingHandle;
#else
    // Linux
    int fileHandle;
#endif

public:
    MappedFile();
    ~MappedFile();

    void open(const std::string& filename, Access access);
    void create(const std::string& filename, size_t getSize);
    void close();

    bool isOpen() const
    {
        return data_ != 0;
    }

    const void* getData() const
    {
        return data_;
    }

    void* getData()
    {
        return data_;
    }

    size_t getSize() const
    {
        return size_;
    }

    void resize(size_t getSize);

private:
    void init();
    void cleanup();
    void map();
    void unmap();
    void setFileSize(size_t getSize);
};

} // namespace prt
