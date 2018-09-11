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

#include <cstdio>
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

void savePoi(const std::string& filename, const Poi& poi)
{
    FILE* file = fopen(filename.c_str(), "at");
    if (file == 0)
    {
        LogError() << "Could not save POI set: " << filename;
        return;
    }

    fprintf(file, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", poi.p.x, poi.p.y, poi.p.z, poi.N.x, poi.N.y, poi.N.z, poi.dist);
    fclose(file);
}

void loadPoiSet(const std::string& filename, Array<Poi>& poiSet)
{
    Log() << "Loading POI set: " << filename;

    FILE* file = fopen(filename.c_str(), "rt");
    if (file == 0)
        return; // silent

    poiSet.clear();
    char line[2048];

    while (!feof(file))
    {
        Poi poi;
        poi.N = zero;
        poi.dist = posInf;

        fgets(line, sizeof(line), file);
        int n = sscanf(line, "%f %f %f %f %f %f %f", &poi.p.x, &poi.p.y, &poi.p.z, &poi.N.x, &poi.N.y, &poi.N.z, &poi.dist);
        if (n == EOF || n == 0)
            break;

        if (n != 3 && n != 6 && n != 7)
        {
            LogError() << "Error loading POI set";
            return;
        }

        poiSet.pushBack(poi);
    }

    fclose(file);

    Log() << "Loaded " << poiSet.getSize() << " POIs";
}

} // namespace prt

