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

#include "vec3.h"

namespace prt {

template <class T>
struct Basis3
{
    Vec3<T> U;
    Vec3<T> V;
    Vec3<T> N;

    FORCEINLINE Basis3() {}

    FORCEINLINE Basis3(One)
        : U(one, zero, zero),
          V(zero, one, zero),
          N(zero, zero, one) {}

    FORCEINLINE Basis3(const Basis3<T>& b) : U(b.U), V(b.V), N(b.N) {}

    template <class T1>
    FORCEINLINE Basis3(const Basis3<T1>& b) : U(b.U), V(b.V), N(b.N) {}

    FORCEINLINE Basis3(const Vec3<T>& U, const Vec3<T>& V, const Vec3<T>& N)
        : U(U), V(V), N(N) {}

    FORCEINLINE Basis3<T>& operator =(const Basis3<T>& b)
    {
        U = b.U;
        V = b.V;
        N = b.N;
        return *this;
    }

    FORCEINLINE Vec3<T> toGlobal(const Vec3<T>& a) const
    {
        return a.x * U + a.y * V + a.z * N;
    }

    FORCEINLINE Vec3<T> toLocal(const Vec3<T>& a) const
    {
        return Vec3<T>(dot(a, U), dot(a, V), dot(a, N));
    }

    FORCEINLINE Basis3<T> rotateU(T angle) const
    {
        Basis3<T> res;
        res.V = rotate(V, U, angle);
        res.N = cross(U, res.V);
        res.U = U;
        return res;
    }

    FORCEINLINE Basis3<T> rotateV(T angle) const
    {
        Basis3<T> res;
        res.N = rotate(N, V, angle);
        res.U = cross(V, res.N);
        res.V = V;
        return res;
    }

    FORCEINLINE Basis3<T> rotateN(T angle) const
    {
        Basis3<T> res;
        res.U = rotate(U, N, angle);
        res.V = cross(N, res.U);
        res.N = N;
        return res;
    }
};

// Typedefs
// --------

typedef Basis3<float> Basis3f;

template <class T, int N>
using Basis3v = Basis3<var<T,N>>;

typedef Basis3<vfloat>   Basis3vf;
typedef Basis3<vfloat4>  Basis3vf4;
typedef Basis3<vfloat8>  Basis3vf8;
typedef Basis3<vfloat16> Basis3vf16;

// Operators
// ---------

template <class T>
FORCEINLINE bool operator ==(const Basis3<T>& a, const Basis3<T>& b)
{
    return a.U == b.U && a.V == b.V && a.N == b.N;
}

template <class T>
FORCEINLINE bool operator !=(const Basis3<T>& a, const Basis3<T>& b)
{
    return a.U != b.U || a.V != b.V || a.N != b.N;
}

template <class T>
FORCEINLINE Vec3<T> operator *(const Basis3<T>& basis, const Vec3<T>& a)
{
    return a.x * basis.U + a.y * basis.V + a.z * basis.N;
}

// Functions
// ---------

// w must be normalized!
template <class T>
FORCEINLINE void makeFrame(Vec3<T>& U, Vec3<T>& V, const Vec3<T>& N)
{
    Vec3<T> U0 = Vec3<T>(zero, N.z, -N.y);
    Vec3<T> U1 = Vec3<T>(-N.z, zero, N.x);
    U = normalize(select(lengthSqr(U0) > lengthSqr(U1), U0, U1));
    V = normalize(cross(N, U));
}

template <class T>
FORCEINLINE Basis3<T> makeFrame(const Vec3<T>& N)
{
    Basis3<T> frame;
    makeFrame(frame.U, frame.V, N);
    frame.N = N;
    return frame;
}

// Stream operators
// ----------------

template <class T>
FORCEINLINE std::ostream& operator <<(std::ostream& osm, const Basis3<T>& a)
{
    osm << a.U << ";" << a.V << ";" << a.N;
    return osm;
}

} // namespace prt
