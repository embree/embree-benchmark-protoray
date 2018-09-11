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

#include "math/vec3.h"
#include "math/vec4.h"

namespace prt {

template <class T>
FORCEINLINE T average(const Vec3<T>& c)
{
    return (c.x + c.y + c.z) * (1.0f/3.0f);
}

// Linear RGB to luminance
template <class T>
FORCEINLINE T luminance(const Vec3<T>& c)
{
    return 0.212671f*c.x + 0.715160f*c.y + 0.072169f*c.z;
}

} // namespace prt
