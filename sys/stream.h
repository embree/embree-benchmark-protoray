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

#include "common.h"

namespace prt {

class Stream
{
public:
    virtual ~Stream() {}

    virtual size_t read(void* dest, size_t count) = 0;
    virtual void write(const void* src, size_t count) = 0;

    void readFull(void* dest, size_t count)
    {
        if (read(dest, count) != count)
            throw std::runtime_error("unexpected end of stream");
    }

    template <class T>
    FORCEINLINE Stream& operator <<(const T& obj)
    {
        static_assert(std::is_trivially_destructible<T>::value, "serialization not implemented");
        write(&obj, sizeof(obj));
        return *this;
    }

    template <class T>
    FORCEINLINE Stream& operator >>(T& obj)
    {
        static_assert(std::is_trivially_destructible<T>::value, "serialization not implemented");
        readFull(&obj, sizeof(obj));
        return *this;
    }
};

} // namespace prt
