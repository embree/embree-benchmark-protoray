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

#include <fstream>
#include "logging.h"
#include "file.h"

namespace prt {

File::File()
    : handle(0)
{
}

File::File(const std::string& filename, FileMode mode)
    : handle(0)
{
    open(filename, mode);
}

File::~File()
{
    if (handle)
        close();
}

void File::open(const std::string& filename, FileMode mode)
{
    if (handle)
        throw std::logic_error("file is already open");

    const char* modeStr = 0;
    switch (mode)
    {
    case fileModeOpen:     modeStr = "rb";  break;
    case fileModeOpenRw:   modeStr = "r+b"; break;
    case fileModeCreate:   modeStr = "wb";  break;
    case fileModeCreateRw: modeStr = "w+b"; break;
    case fileModeAppend:   modeStr = "ab";  break;
    case fileModeAppendRw: modeStr = "a+b"; break;

    default:
        throw std::invalid_argument("invalid file mode");
    }

    handle = fopen(filename.c_str(), modeStr);
    if (!handle)
        throw std::runtime_error("could not open file: " + filename);
}

void File::create(const std::string& filename)
{
    open(filename, fileModeCreate);
}

void File::close()
{
    if (!handle)
        throw std::logic_error("file is not open");

    if (fclose(handle) != 0)
        throw std::runtime_error("fclose failed");

    handle = 0;
}

bool File::exists(const std::string& filename)
{
    // FIXME
    std::ifstream is(filename.c_str());
    return (bool)is;
}

void File::remove(const std::string& filename)
{
    std::remove(filename.c_str());
}

} // namespace prt
