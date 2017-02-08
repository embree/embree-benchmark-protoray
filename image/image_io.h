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

#include "sys/common.h"
#include "math/vec2.h"
#include "pixel.h"

namespace prt {

// Low-level image I/O
class ImageIo
{
public:
    struct Desc
    {
        Vec2i size;
        PixelFormat format;
    };

    struct HandleSt;
    typedef HandleSt* Handle;

    static Handle open(const std::string& filename);
    static Desc getDesc(Handle h);
    static void getData(Handle h, void* data);
    static void close(Handle h);

    static bool save(const std::string& filename, const Desc& desc, const void* data);
};

} // namespace prt
