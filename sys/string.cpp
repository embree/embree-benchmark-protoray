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

#include <cmath>
#include "string.h"

namespace prt {

Stream& operator <<(Stream& osm, const char* str)
{
    int size = (int)strlen(str);
    osm << size;
    osm.write(str, size);
    return osm;
}

Stream& operator <<(Stream& osm, const std::string& str)
{
    int size = (int)str.size();
    osm << size;
    osm.write(&str.front(), size);
    return osm;
}

Stream& operator >>(Stream& ism, std::string& str)
{
    int size;
    ism >> size;
    str.resize(size);
    ism.readFull(&str.front(), size);
    return ism;
}

} // namespace prt
