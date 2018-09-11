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
#include "string.h"
#include "stream.h"
#include "constants.h"
#include "math/vec2.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "math/basis3.h"

namespace prt {

// Variant value
class Value
{
public:
    enum Type
    {
        typeEmpty,
        typeInt1,
        typeInt2,
        typeInt3,
        typeFloat1,
        typeFloat2,
        typeFloat3,
        typeBasis3,
        typeLong1,
        typeDouble1,
        typeString
    };

private:
    Type type;

    union
    {
        int i[9];
        float f[9];
        int64_t l[3];
        double d[3];
    } num;
    std::string str;

public:
    Value() : type(typeEmpty) {}
    Value(Empty) : type(typeEmpty) {}

    Value(int a) : type(typeInt1) { num.i[0] = a; }

    Value(const Vec2i& a) : type(typeInt2)
    {
        num.i[0] = a[0];
        num.i[1] = a[1];
    }

    Value(const Vec3i& a) : type(typeInt3)
    {
        num.i[0] = a[0];
        num.i[1] = a[1];
        num.i[2] = a[2];
    }

    Value(float a) : type(typeFloat1) { num.f[0] = a; }

    Value(const Vec2f& a) : type(typeFloat2)
    {
        num.f[0] = a[0];
        num.f[1] = a[1];
    }

    Value(const Vec3f& a) : type(typeFloat3)
    {
        num.f[0] = a[0];
        num.f[1] = a[1];
        num.f[2] = a[2];
    }

    Value(const Basis3f& a) : type(typeBasis3)
    {
        num.f[0] = a.U.x; num.f[1] = a.U.y; num.f[2] = a.U.z;
        num.f[3] = a.V.x; num.f[4] = a.V.y; num.f[5] = a.V.z;
        num.f[6] = a.N.x; num.f[7] = a.N.y; num.f[8] = a.N.z;
    }

    Value(int64_t a) : type(typeLong1) { num.l[0] = a; }

    Value(double a) : type(typeDouble1) { num.d[0] = a; }

    Value(const char* s) : type(typeString) { str = s; }
    Value(const std::string& s) : type(typeString) { str = s; }

    operator std::string() const;

    bool isEmpty() const
    {
        return type == typeEmpty;
    }

    Type getType() const
    {
        return type;
    }

    template <class T>
    T get() const;

    friend std::ostream& operator <<(std::ostream& osm, const Value& val);

    template <class StreamType>
    friend void serialize(StreamType& osm, const Value& val);

    template <class StreamType>
    friend void deserialize(StreamType& ism, Value& val);
};

template <>
inline int Value::get() const
{
    switch (type)
    {
    case typeInt1:
        return num.i[0];

    case typeFloat1:
        return (int)num.f[0];

    case typeLong1:
        return (int)num.l[0];

    case typeDouble1:
        return (int)num.d[0];

    case typeString:
        return fromString<int>(str);

    default:
        throw std::logic_error("incompatible type");
    }
}

template <>
inline Vec2i Value::get() const
{
    switch (type)
    {
    case typeInt1:
        return Vec2i(num.i[0]);

    case typeFloat1:
        return Vec2i((int)num.f[0]);

    case typeInt2:
        return Vec2i(num.i[0], num.i[1]);

    case typeFloat2:
        return Vec2i((int)num.f[0], (int)num.f[1]);

    case typeString:
        return fromString<Vec2i>(str);

    default:
        throw std::logic_error("incompatible type");
    }
}

template <>
inline Vec3i Value::get() const
{
    switch (type)
    {
    case typeInt1:
        return Vec3i(num.i[0]);

    case typeFloat1:
        return Vec3i((int)num.f[0]);

    case typeInt3:
        return Vec3i(num.i[0], num.i[1], num.i[2]);

    case typeFloat3:
        return Vec3i((int)num.f[0], (int)num.f[1], (int)num.f[2]);

    case typeString:
        return fromString<Vec3i>(str);

    default:
        throw std::logic_error("incompatible type");
    }
}

template <>
inline float Value::get() const
{
    switch (type)
    {
    case typeInt1:
        return (float)num.i[0];

    case typeFloat1:
        return num.f[0];

    case typeLong1:
        return (float)num.l[0];

    case typeDouble1:
        return (float)num.d[0];

    case typeString:
        return fromString<float>(str);

    default:
        throw std::logic_error("incompatible type");
    }
}

template <>
inline Vec2f Value::get() const
{
    switch (type)
    {
    case typeInt1:
        return Vec2f((float)num.i[0]);

    case typeFloat1:
        return Vec2f(num.f[0]);

    case typeInt2:
        return Vec2f((float)num.i[0], (float)num.i[1]);

    case typeFloat2:
        return Vec2f(num.f[0], num.f[1]);

    case typeString:
        return fromString<Vec2f>(str);

    default:
        throw std::logic_error("incompatible type");
    }
}

template <>
inline Vec3f Value::get() const
{
    switch (type)
    {
    case typeInt1:
        return Vec3f((float)num.i[0]);

    case typeFloat1:
        return Vec3f(num.f[0]);

    case typeInt3:
        return Vec3f((float)num.i[0], (float)num.i[1], (float)num.i[2]);

    case typeFloat3:
        return Vec3f(num.f[0], num.f[1], num.f[2]);

    case typeString:
        return fromString<Vec3f>(str);

    default:
        throw std::logic_error("incompatible type");
    }
}

template <>
inline Basis3f Value::get() const
{
    switch (type)
    {
    case typeBasis3:
        return Basis3f(Vec3f(num.f[0], num.f[1], num.f[2]),
                       Vec3f(num.f[3], num.f[4], num.f[5]),
                       Vec3f(num.f[6], num.f[7], num.f[8]));

    //case typeString:
    //    return fromString<Basis3f>(str);

    default:
        throw std::logic_error("incompatible type");
    }
}

template <>
inline int64_t Value::get() const
{
    switch (type)
    {
    case typeInt1:
        return num.i[0];

    case typeFloat1:
        return num.f[0];

    case typeLong1:
        return num.l[0];

    case typeDouble1:
        return num.d[0];

    case typeString:
        return fromString<double>(str);

    default:
        throw std::logic_error("incompatible type");
    }
}

template <>
inline double Value::get() const
{
    switch (type)
    {
    case typeInt1:
        return num.i[0];

    case typeFloat1:
        return num.f[0];

    case typeLong1:
        return (double)num.l[0];

    case typeDouble1:
        return num.d[0];

    case typeString:
        return fromString<double>(str);

    default:
        throw std::logic_error("incompatible type");
    }
}

template <>
inline std::string Value::get() const
{
    if (type == typeEmpty)
        return std::string();

    if (type == typeString)
        return str;

    std::stringstream sm;
    sm << *this;
    return sm.str();
}

inline Value::operator std::string() const
{
    return get<std::string>();
}

inline std::ostream& operator <<(std::ostream& osm, const Value& val)
{
    switch (val.type)
    {
    case Value::typeInt1:
        return osm << val.num.i[0];

    case Value::typeInt2:
        return osm << Vec2i(val.num.i[0], val.num.i[1]);

    case Value::typeInt3:
        return osm << Vec3i(val.num.i[0], val.num.i[1], val.num.i[2]);

    case Value::typeFloat1:
        return osm << val.num.f[0];

    case Value::typeFloat2:
        return osm << Vec2f(val.num.f[0], val.num.f[1]);

    case Value::typeFloat3:
        return osm << Vec3f(val.num.f[0], val.num.f[1], val.num.f[2]);

    case Value::typeBasis3:
        return osm << Basis3f(Vec3f(val.num.f[0], val.num.f[1], val.num.f[2]),
                              Vec3f(val.num.f[3], val.num.f[4], val.num.f[5]),
                              Vec3f(val.num.f[6], val.num.f[7], val.num.f[8]));

    case Value::typeLong1:
        return osm << val.num.l[0];

    case Value::typeDouble1:
        return osm << val.num.d[0];

    case Value::typeString:
        return osm << val.str;

    default:
        return osm;
    }
}

// Serialization
// -------------

template <class StreamType>
void serialize(StreamType& osm, const Value& val)
{
    osm << val.type;

    switch (val.type)
    {
    case Value::typeEmpty:
        break;

    case Value::typeString:
        osm << val.str;
        break;

    default:
        osm << val.num;
        break;
    }
}

template <class StreamType>
void deserialize(StreamType& ism, Value& val)
{
    ism >> val.type;

    switch (val.type)
    {
    case Value::typeEmpty:
        break;

    case Value::typeString:
        ism >> val.str;
        break;

    default:
        val.str.clear();
        ism >> val.num;
        break;
    }
}

inline Stream& operator <<(Stream& osm, const Value& val) { serialize(osm, val); return osm; }
inline Stream& operator >>(Stream& ism, Value& val) { deserialize(ism, val); return ism; }

} // namespace prt
