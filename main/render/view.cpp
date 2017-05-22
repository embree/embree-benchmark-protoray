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

#include "sys/logging.h"
#include "view.h"

namespace prt {

void saveViewSet(const std::string& filename, const ViewSet& viewSet)
{
    FILE* file = fopen(filename.c_str(), "wb");
    if (file == 0)
    {
        LogError() << "Could not save view set: " << filename;
        return;
    }

    fwrite(&viewSet, sizeof(ViewSet), 1, file);
    fclose(file);
}

void loadViewSet(const std::string& filename, ViewSet& viewSet)
{
    Log() << "Loading view set: " << filename;

    FILE* file = fopen(filename.c_str(), "rb");
    if (file == 0)
        return; // silent

    fread(&viewSet, sizeof(ViewSet), 1, file);
    fclose(file);
}

} // namespace prt

