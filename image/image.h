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
#include "math/vec2.h"
#include "math/vec3.h"
#include "math/vec4.h"

namespace prt {

template <class T>
class Image
{
private:
    T* data;
    Vec2i size;

public:
    FORCEINLINE Image() : data(0), size(0) {}

    explicit Image(const Vec2i& size)
    {
        init(size);
    }

    Image(int width, int height)
    {
        init(Vec2i(width, height));
    }

    Image(const Image& other)
    {
        init(other.size);
        memcpy(data, other.data, size.x * size.y * sizeof(T));
    }

    Image& operator =(const Image& other)
    {
        alloc(other.size);
        memcpy(data, other.data, size.x * size.y * sizeof(T));
        return *this;
    }

    ~Image()
    {
        free();
    }

    void alloc(const Vec2i& size)
    {
        free();
        init(size);
    }

    void alloc(int width, int height)
    {
        alloc(Vec2i(width, height));
    }

    void free()
    {
        alignedFree(data);
    }

    FORCEINLINE const T* operator [](int y) const
    {
        assert(y >= 0 && y < size.y);
        return data + y * size.x;
    }

    FORCEINLINE T* operator [](int y)
    {
        assert(y >= 0 && y < size.y);
        return data + y * size.x;
    }

    FORCEINLINE const T* getData() const
    {
        return data;
    }

    FORCEINLINE T* getData()
    {
        return data;
    }

    FORCEINLINE Vec2i getSize() const
    {
        return size;
    }

    FORCEINLINE int getWidth() const
    {
        return size.x;
    }

    FORCEINLINE int getHeight() const
    {
        return size.y;
    }

private:
    void init(const Vec2i& size)
    {
        data = (T*)alignedAlloc(size.x * size.y * sizeof(T));
        this->size = size;
    }
};

typedef Image<Vec4uc> Image4uc; // BGRA uint8
typedef Image<Vec3f> Image3f;   // RGB float
typedef Image<float> Image1f;   // float

bool loadImage(const std::string& filename, Image4uc& image);
bool loadImage(const std::string& filename, Image3f& image);

bool saveImage(const std::string& filename, const Image4uc& image);
bool saveImage(const std::string& filename, const Image<int>& image);
bool saveImage(const std::string& filename, const Image3f& image);
bool saveImage(const std::string& filename, const Image1f& image);

} // namespace prt
