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

#include "sys/common.h"
#include "sys/array.h"
#include "math/vec3.h"

namespace prt {

struct View
{
    Vec3f pos;
    float angleX;
    float angleY;
    float fovY;
    float radius; // lens radius
    float focus;  // focal distance
};

struct ViewSet
{
    static const int size = 10;

    View views[size];
};

struct Poi
{
    Vec3f p;    // position
    Vec3f N;    // normal
    float dist; // distance

    Poi() {}
    Poi(Zero) : p(zero), N(zero), dist(posInf) {}
    Poi(const Vec3f& p, const Vec3f& N, float dist) : p(p), N(N), dist(dist) {}
};

void saveViewSet(const std::string& filename, const ViewSet& viewSet);
void loadViewSet(const std::string& filename, ViewSet& viewSet);

void savePoi(const std::string& filename, const Poi& poi);
void loadPoiSet(const std::string& filename, Array<Poi>& poiSet);

} // namespace prt
