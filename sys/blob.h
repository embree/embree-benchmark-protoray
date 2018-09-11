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
#include "logging.h"
#include "array.h"
#include "file.h"

namespace prt {

template <class T>
inline void loadBlob(const std::string& filename, T& obj)
{
    Log() << "Loading blob: " << filename;

    File file;
    file.open(filename);
    int id;
    file >> id;
    if (id != T::blobId)
        throw std::runtime_error("file has wrong blob ID: " + filename);
    file >> obj;
}

template <class T>
inline void saveBlob(const std::string& filename, const T& obj)
{
    Log() << "Saving blob: " << filename;

    File file;
    file.create(filename);
    int id = T::blobId;
    file << id;
    file << obj;
}

} // namespace prt
