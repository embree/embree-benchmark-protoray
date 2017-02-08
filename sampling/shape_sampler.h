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
#include "math/vec3.h"
#include "math/basis3.h"

namespace prt {

template <class T>
FORCEINLINE Vec3<T> uniformSampleSphere(T& pdf, const Vec2<T>& s)
{
    T cosTheta = 1.0f - 2.0f*s.x;
    T sinTheta = 2.0f * sqrt(s.x * (1.0f-s.x));

    T phi = 2.0f*float(pi) * s.y;

    pdf = 1.0f/(4.0f*float(pi));

    T sinPhi, cosPhi;
    sincos(phi, sinPhi, cosPhi);

    T x = cosPhi * sinTheta;
    T y = sinPhi * sinTheta;
    T z = cosTheta;
    return Vec3<T>(x,y,z);
}

FORCEINLINE float uniformSampleSpherePdf()
{
    return 1.0f/(4.0f*float(pi));
}

template <class T>
FORCEINLINE Vec3<T> cosineSampleHemisphere(T& pdf, const Vec2<T>& s)
{
    T cosTheta = sqrt(s.x);
    T sinTheta = sqrt(1.0f-s.x);

    T phi = 2.0f*float(pi) * s.y;

    pdf = cosTheta * (1.0f/float(pi));

    T sinPhi, cosPhi;
    sincos(phi, sinPhi, cosPhi);

    T x = cosPhi * sinTheta;
    T y = sinPhi * sinTheta;
    T z = cosTheta;
    return Vec3<T>(x,y,z);
}

template <class T>
FORCEINLINE Vec3<T> cosineSampleHemisphere(const Vec2<T>& s)
{
    T pdf;
    return cosineSampleHemisphere(pdf, s);
}

template <class T>
FORCEINLINE T cosineSampleHemispherePdf(T cosTheta)
{
    return max(cosTheta * (1.0f/float(pi)), 0.0f);
}

template <class T>
FORCEINLINE Vec3<T> uniformSampleCone(T& pdf, const Vec2<T>& s, T angle)
{
    T cosTheta = 1.0f - s.x*(1.0f-cos(angle));
    T sinTheta = cos2sin(cosTheta);

    T phi = 2.0f*float(pi) * s.y;

    pdf = rcp(4.0f*float(pi)*sqr(sin(0.5f*angle)));

    T sinPhi, cosPhi;
    sincos(phi, sinPhi, cosPhi);

    T x = cosPhi * sinTheta;
    T y = sinPhi * sinTheta;
    T z = cosTheta;
    return Vec3<T>(x,y,z);
}

template <class T>
FORCEINLINE T uniformSampleConePdf(T cosTheta, T angle)
{
    return select(cosTheta < cos(angle), T(0.0f), rcp(4.0f*float(pi)*sqr(sin(0.5f*angle))));
}

template <class T>
FORCEINLINE Vec3<T> uniformSampleTriangle(const Vec2<T>& s, const Vec3<T>& v0, const Vec3<T>& v1, const Vec3<T>& v2)
{
    T rx = sqrt(s.x);
    return v2 + (1.0f-rx)*(v0-v2) + (s.y*rx)*(v1-v2);
}

template <class T>
FORCEINLINE Vec2<T> uniformSampleDisk(const Vec2<T>& s)
{
    T r = sqrt(s.x);
    T theta = 2.0f*float(pi) * s.y;

    T sinTheta, cosTheta;
    sincos(theta, sinTheta, cosTheta);

    return Vec2<T>(r*cosTheta, r*sinTheta);
}

} // namespace prt
