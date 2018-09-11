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
#include "atomic.h"
#include "memory.h"

namespace prt {

struct ALIGNED_CACHE RefHeader
{
    Atomic<uint32_t> refCount; // reference count
    void* data;                // pointer to the allocated data
};

template <class T>
class ref
{
private:
    T* ptr;

public:
    template <class T1>
    friend class ref;

    template <class T1, class... Args>
    friend ref<T1> makeRef(Args&&... args);

    template <class T1, class T>
    friend ref<T1> staticRefCast(const ref<T>& r);

    template <class T1, class T>
    friend ref<T1> dynamicRefCast(const ref<T>& r);

    FORCEINLINE ref() : ptr(0) {}

    FORCEINLINE ref(std::nullptr_t) : ptr(0) {}

    FORCEINLINE ref(const ref<T>& other) : ptr(other.ptr)
    {
        incRef();
    }

    template <class T1>
    FORCEINLINE ref(const ref<T1>& other) : ptr(other.ptr)
    {
        static_assert((T*)1 == (T*)(T1*)1, "ref conversion not supported");
        incRef();
    }

    FORCEINLINE ref(ref<T>&& other) : ptr(other.ptr)
    {
        other.ptr = 0;
    }

    template <class T1>
    FORCEINLINE ref(ref<T1>&& other) : ptr(other.ptr)
    {
        static_assert((T*)1 == (T*)(T1*)1, "ref conversion not supported");
        other.ptr = 0;
    }

    FORCEINLINE ~ref()
    {
        decRef();
    }

    FORCEINLINE ref<T>& operator =(const ref<T>& other)
    {
        other.incRef();
        decRef();
        ptr = other.ptr;
        return *this;
    }

    template <class T1>
    FORCEINLINE ref<T>& operator =(const ref<T1>& other)
    {
        static_assert((T*)1 == (T*)(T1*)1, "ref conversion not supported");
        other.incRef();
        decRef();
        ptr = other.ptr;
        return *this;
    }

    FORCEINLINE ref<T>& operator =(ref<T>&& other)
    {
        ref<T>(std::move(other)).swap(*this);
        return *this;
    }

    template <class T1>
    FORCEINLINE ref<T>& operator =(ref<T1>&& other)
    {
        static_assert((T*)1 == (T*)(T1*)1, "ref conversion not supported");
        ref<T>(std::move(other)).swap(*this);
        return *this;
    }

    FORCEINLINE operator bool() const
    {
        return ptr != 0;
    }

    FORCEINLINE void reset()
    {
        decRef();
        ptr = 0;
    }

    FORCEINLINE void swap(ref<T>& other)
    {
        std::swap(ptr, other.ptr);
    }

    FORCEINLINE T& operator *() const { return *ptr; }
    FORCEINLINE T* operator ->() const { return ptr; }
    FORCEINLINE T* get() const { return ptr; }

private:
    FORCEINLINE ref(T* ptr, void* data) : ptr(ptr)
    {
        RefHeader* hdr = getHeader();
        hdr->refCount = 1;
        hdr->data = data;
    }

    FORCEINLINE RefHeader* getHeader() const
    {
        // The header is just before the object
        return (RefHeader*)ptr - 1;
    }

    FORCEINLINE void incRef() const
    {
        if (ptr)
            getHeader()->refCount++;
    }

    FORCEINLINE void decRef()
    {
        if (ptr)
        {
            RefHeader* hdr = getHeader();
            if (hdr->refCount-- == 1)
            {
                void* data = hdr->data;
                ptr->~T();
                alignedFree(data);
            }
        }
    }
};

template <class T, class... Args>
FORCEINLINE ref<T> makeRef(Args&&... args)
{
    // Dummy union aligned for both the header and the object
    union ObjData
    {
        RefHeader hdr;
        typename std::aligned_storage<sizeof(T), std::alignment_of<T>::value>::type obj;
    };

    // Dummy structure to get the allocation size, alignment, and object offset
    struct Data
    {
        RefHeader hdr;
        typename std::aligned_storage<sizeof(T), std::alignment_of<ObjData>::value>::type obj;
    };

    void* data = alignedAlloc(sizeof(Data), std::alignment_of<Data>::value);

    size_t offset = offsetof(Data, obj);
    T* ptr = (T*)((char*)data + offset);

    try
    {
        new (ptr) T(std::forward<Args>(args)...);
    }
    catch (...)
    {
        alignedFree(data);
        throw;
    }

    return ref<T>(ptr, data);
}

template <class T1, class T>
FORCEINLINE ref<T1> staticRefCast(const ref<T>& r)
{
    ref<T1> r1;
    r1.ptr = static_cast<T1*>(r.ptr);
    r1.incRef();
    return r1;
}

template <class T1, class T>
FORCEINLINE ref<T1> dynamicRefCast(const ref<T>& r)
{
    ref<T1> r1;
    r1.ptr = dynamic_cast<T1*>(r.ptr);
    r1.incRef();
    return r1;
}

} // namespace prt
