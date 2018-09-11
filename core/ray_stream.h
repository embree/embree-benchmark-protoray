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

#include "sys/memory.h"
#include "ray_simd.h"

namespace prt {

struct RayStreamIdN;
struct RayStreamIdN2;

#if defined(__AVX512F__) || defined(__KNC__)
// AVX-512 & KNC
typedef RayStreamIdN2 RayStreamId;
const int defaultRayStreamSize = 4096;
#else
// AVX2
typedef RayStreamIdN RayStreamId;
const int defaultRayStreamSize = 2048;
#endif

const int rayStreamMsTileX = 16;
const int rayStreamMsTileY = 16;

const int rayStreamPfL2Dist = 2*cacheLineSize; // prefetch into L2 distance

template <class T, int size>
class RayStreamChannel
{
private:
    ALIGNED_CACHE T v[size + simdSize]; // padded

public:
    FORCEINLINE void setA(int i, const var<T>& a)
    {
    #ifdef __KNC__
        prefetchL1Ex(&v[i]);
    #endif
        prefetchL2Ex(&v[i], rayStreamPfL2Dist);

        store(&v[i], a);
    }

    FORCEINLINE void set(int i, const var<T>& a)
    {
    #ifdef __KNC__
        prefetch2L1Ex(&v[i]);
    #endif
        prefetchL2Ex(&v[i], rayStreamPfL2Dist);

        ustore(&v[i], a);
    }

    FORCEINLINE void packSet(vbool m, int i, const var<T>& a)
    {
    #ifdef __KNC__
        prefetch2L1Ex(&v[i]);
    #endif
        prefetchL2Ex(&v[i], rayStreamPfL2Dist);

        ucompress(m, &v[i], a);
    }

    FORCEINLINE var<T> getA(int i) const
    {
    #ifdef __KNC__
        prefetchL1(&v[i]);
    #endif
        prefetchL2(&v[i], rayStreamPfL2Dist);

        return load(&v[i]);
    }

    FORCEINLINE var<T> get(int i) const
    {
    #ifdef __KNC__
        prefetch2L1(&v[i]);
    #endif
        prefetchL2(&v[i], rayStreamPfL2Dist);

        return uload(&v[i]);
    }

    FORCEINLINE var<T> get(vbool m, vint i) const
    {
    #ifdef __KNC__
        prefetchGatherL1(m, v, i);
    #endif

        return gather(m, v, i);
    }

    FORCEINLINE var<T> get(vbool m, const RayStreamIdN& i) const
    {
        return get(m, i.cur);
    }

    FORCEINLINE var<T> get(vbool m, const RayStreamIdN2& i) const
    {
        prefetchGatherL2(v, i.next);
        return get(m, i.cur);
    }

    FORCEINLINE var<T> getUnpack(vbool m, int i) const
    {
    #ifdef __KNC__
        prefetch2L1(&v[i]);
    #endif
        prefetchL2(&v[i], rayStreamPfL2Dist);

        return uexpand(m, &v[i]);
    }

    FORCEINLINE const T& operator [](size_t i) const
    {
        return v[i];
    }

    FORCEINLINE T& operator [](size_t i)
    {
        return v[i];
    }

    FORCEINLINE const T* get() const { return v; }
    FORCEINLINE T* get() { return v; }
};

template <class T, int size>
class RayStreamChannel3
{
private:
    typedef T V[size + simdSize]; // padded

public:
    ALIGNED_CACHE V x;
    ALIGNED_CACHE V y;
    ALIGNED_CACHE V z;

    FORCEINLINE void setA(int i, const Vec3v<T>& a)
    {
    #ifdef __KNC__
        prefetchL1Ex(&x[i]);
        prefetchL1Ex(&y[i]);
        prefetchL1Ex(&z[i]);
    #endif

        prefetchL2Ex(&x[i], rayStreamPfL2Dist);
        prefetchL2Ex(&y[i], rayStreamPfL2Dist);
        prefetchL2Ex(&z[i], rayStreamPfL2Dist);

        store(&x[i], a[0]);
        store(&y[i], a[1]);
        store(&z[i], a[2]);
    }

    FORCEINLINE void set(int i, const Vec3v<T>& a)
    {
    #ifdef __KNC__
        prefetch2L1Ex(&x[i]);
        prefetch2L1Ex(&y[i]);
        prefetch2L1Ex(&z[i]);
    #endif

        prefetchL2Ex(&x[i], rayStreamPfL2Dist);
        prefetchL2Ex(&y[i], rayStreamPfL2Dist);
        prefetchL2Ex(&z[i], rayStreamPfL2Dist);

        ustore(&x[i], a[0]);
        ustore(&y[i], a[1]);
        ustore(&z[i], a[2]);
    }

    FORCEINLINE void packSet(vbool m, int i, const Vec3v<T>& a)
    {
    #ifdef __KNC__
        prefetch2L1Ex(&x[i]);
        prefetch2L1Ex(&y[i]);
        prefetch2L1Ex(&z[i]);
    #endif

        prefetchL2Ex(&x[i], rayStreamPfL2Dist);
        prefetchL2Ex(&y[i], rayStreamPfL2Dist);
        prefetchL2Ex(&z[i], rayStreamPfL2Dist);

        ucompress(m, &x[i], a[0]);
        ucompress(m, &y[i], a[1]);
        ucompress(m, &z[i], a[2]);
    }

    FORCEINLINE Vec3v<T> getA(int i) const
    {
    #ifdef __KNC__
        prefetchL1(&x[i]);
        prefetchL1(&y[i]);
        prefetchL1(&z[i]);
    #endif

        prefetchL2(&x[i], rayStreamPfL2Dist);
        prefetchL2(&y[i], rayStreamPfL2Dist);
        prefetchL2(&z[i], rayStreamPfL2Dist);

        return Vec3v<T>(load(&x[i]),
                        load(&y[i]),
                        load(&z[i]));
    }

    FORCEINLINE Vec3v<T> get(int i) const
    {
    #ifdef __KNC__
        prefetch2L1(&x[i]);
        prefetch2L1(&y[i]);
        prefetch2L1(&z[i]);
    #endif

        prefetchL2(&x[i], rayStreamPfL2Dist);
        prefetchL2(&y[i], rayStreamPfL2Dist);
        prefetchL2(&z[i], rayStreamPfL2Dist);

        return Vec3v<T>(uload(&x[i]),
                        uload(&y[i]),
                        uload(&z[i]));
    }

    FORCEINLINE Vec3v<T> get(vbool m, vint i) const
    {
    #ifdef __KNC__
        prefetchGatherL1(m, x, i);
        prefetchGatherL1(m, y, i);
        prefetchGatherL1(m, z, i);
    #endif

        return Vec3v<T>(gather(m, x, i),
                        gather(m, y, i),
                        gather(m, z, i));
    }

    FORCEINLINE Vec3v<T> get(vbool m, const RayStreamIdN& i) const
    {
        return get(m, i.cur);
    }

    FORCEINLINE Vec3v<T> get(vbool m, const RayStreamIdN2& i) const
    {
        prefetchGatherL2(x, i.next);
        prefetchGatherL2(y, i.next);
        prefetchGatherL2(z, i.next);

        return get(m, i.cur);
    }

    FORCEINLINE Vec3v<T> getUnpack(vbool m, int i) const
    {
    #ifdef __KNC__
        prefetch2L1(&x[i]);
        prefetch2L1(&y[i]);
        prefetch2L1(&z[i]);
    #endif

        prefetchL2(&x[i], rayStreamPfL2Dist);
        prefetchL2(&y[i], rayStreamPfL2Dist);
        prefetchL2(&z[i], rayStreamPfL2Dist);

        return Vec3v<T>(uexpand(m, &x[i]),
                        uexpand(m, &y[i]),
                        uexpand(m, &z[i]));
    }

    FORCEINLINE const T* getX() const { return x; }
    FORCEINLINE const T* getY() const { return y; }
    FORCEINLINE const T* getZ() const { return z; }

    FORCEINLINE T* getX() { return x; }
    FORCEINLINE T* getY() { return y; }
    FORCEINLINE T* getZ() { return z; }
};

struct RayStreamIdN
{
    vint cur;

    FORCEINLINE RayStreamIdN(vint cur) : cur(cur) {}

    template <int size>
    static FORCEINLINE RayStreamIdN get(const RayStreamChannel<int,size>& ids, int i)
    {
        return RayStreamIdN(ids.get(i));
    }

    template <int size>
    static FORCEINLINE RayStreamIdN getA(const RayStreamChannel<int,size>& ids, int i)
    {
        return RayStreamIdN(ids.getA(i));
    }
};

struct RayStreamIdN2
{
    vint cur;
    vint next;

    FORCEINLINE RayStreamIdN2(vint cur, vint next) : cur(cur), next(next) {}

    template <int size>
    static FORCEINLINE RayStreamIdN2 get(const RayStreamChannel<int,size>& ids, int i)
    {
        return RayStreamIdN2(ids.get(i), ids.get(i+simdSize));
    }

    template <int size>
    static FORCEINLINE RayStreamIdN2 getA(const RayStreamChannel<int,size>& ids, int i)
    {
        return RayStreamIdN2(ids.getA(i), ids.getA(i+simdSize));
    }
};

template <int size>
struct RayStream
{
    RayStreamChannel3<float, size> org;
    RayStreamChannel3<float, size> dir;
    RayStreamChannel<float, size> far;

    FORCEINLINE float* getOrgX() { return org.getX(); }
    FORCEINLINE float* getOrgY() { return org.getY(); }
    FORCEINLINE float* getOrgZ() { return org.getZ(); }
    FORCEINLINE float* getDirX() { return dir.getX(); }
    FORCEINLINE float* getDirY() { return dir.getY(); }
    FORCEINLINE float* getDirZ() { return dir.getZ(); }
    FORCEINLINE float* getFar()  { return far.get(); }

    FORCEINLINE void setA(int i, const RaySimd& ray)
    {
        org.setA(i, ray.org);
        dir.setA(i, ray.dir);
        far.setA(i, ray.far);
    }

    FORCEINLINE void set(int i, const RaySimd& ray)
    {
        org.set(i, ray.org);
        dir.set(i, ray.dir);
        far.set(i, ray.far);
    }

    FORCEINLINE void packSet(vbool m, int i, const RaySimd& ray)
    {
        org.packSet(m, i, ray.org);
        dir.packSet(m, i, ray.dir);
        far.packSet(m, i, ray.far);
    }

    FORCEINLINE void getA(int i, RaySimd& ray) const
    {
        ray.org = org.getA(i);
        ray.dir = dir.getA(i);
        ray.far = far.getA(i);
    }

    template <class RayStreamIdT>
    FORCEINLINE void get(vbool m, const RayStreamIdT& i, RaySimd& ray) const
    {
        ray.org = org.get(m, i);
        ray.dir = dir.get(m, i);
        ray.far = far.get(m, i);
    }

    FORCEINLINE bool isHit(int i) const
    {
        return far[i] < float(posMax);
    }
};

template <int size>
struct HitStream
{
    RayStreamChannel<int, size> primId;
    RayStreamChannel<float, size> u;
    RayStreamChannel<float, size> v;

    FORCEINLINE int*   getPrimId() { return primId.get(); }
    FORCEINLINE float* getU()      { return u.get(); }
    FORCEINLINE float* getV()      { return v.get(); }

    FORCEINLINE void getA(int i, HitSimd& hit) const
    {
        hit.primId = primId.getA(i);
        hit.u = u.getA(i);
        hit.v = v.getA(i);
    }

    template <class RayStreamIdT>
    FORCEINLINE void get(vbool m, const RayStreamIdT& i, HitSimd& hit) const
    {
        hit.primId = primId.get(m, i);
        hit.u = u.get(m, i);
        hit.v = v.get(m, i);
    }
};

} // namespace prt
