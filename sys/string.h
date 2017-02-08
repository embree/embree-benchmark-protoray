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

#include <cstring>
#include <string>
#include <sstream>
#include "common.h"
#include "stream.h"

namespace prt {

template <class T>
inline std::string toString(const T& a)
{
    std::stringstream sm;
    sm << a;
    return sm.str();
}

template <class T>
inline std::string toString(const T& a, int precision)
{
    std::stringstream sm;
    sm << std::fixed << std::setprecision(precision) << a;
    return sm.str();
}

template <class T>
inline T fromString(const std::string& str)
{
    std::stringstream sm(str);
    T a;
    sm >> a;
    return a;
}

Stream& operator <<(Stream& osm, const char* str);
Stream& operator <<(Stream& osm, const std::string& str);
Stream& operator >>(Stream& ism, std::string& str);

} // namespace prt
