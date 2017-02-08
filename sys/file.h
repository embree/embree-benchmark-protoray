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
#include "stream.h"

namespace prt {

enum FileMode
{
    fileModeOpen,
    fileModeOpenRw,
    fileModeCreate,
    fileModeCreateRw,
    fileModeAppend,
    fileModeAppendRw
};

class File : public Stream, Uncopyable
{
private:
    FILE* handle;

public:
    File();
    File(const std::string& filename, FileMode mode = fileModeOpen);
    ~File();

    void open(const std::string& filename, FileMode mode = fileModeOpen);
    void create(const std::string& filename);
    void close();

    size_t read(void* dest, size_t count)
    {
        if (!handle)
            throw std::logic_error("file is not open");

        size_t readCount = fread(dest, 1, count, handle);
        return readCount;
    }

    void write(const void* src, size_t count)
    {
        if (!handle)
            throw std::logic_error("file is not open");

        size_t writeCount = fwrite(src, 1, count, handle);
        if (writeCount != count)
            throw std::runtime_error("fwrite failed");
    }

    bool isOpen() const
    {
        return handle != 0;
    }

    static bool exists(const std::string& filename);
    static void remove(const std::string& filename);
};

} // namespace prt
