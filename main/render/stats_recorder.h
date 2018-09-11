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

#include "sys/array.h"
#include "sys/props.h"

namespace prt {

class StatsRecorder
{
private:
    std::vector<std::string> names;
    std::vector<std::vector<double>> values;
    std::map<std::string, size_t> map; // maps names to indices in the values array

public:
    void add(const Props& stats);
    void getAverage(Props& avg);
    void saveCsv(const std::string& filename);
};

} // namespace prt
