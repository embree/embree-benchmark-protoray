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

#include "file.h"
#include "text_reader.h"
#include "option.h"

namespace prt {

std::ostream& operator <<(std::ostream& osm, const Option& opt)
{
    if (!opt.name.empty())
        osm << "-" << opt.name;

    if (!opt.name.empty() && !opt.value.isEmpty())
        osm << "=";

    if (!opt.value.isEmpty())
        osm << opt.value;

    return osm;
}

void parseOptions(int argc, char* argv[], Array<Option>& opts)
{
    char buffer[1024];
    bool isOpt = false;
    Option opt;

    for (int i = 1; i < argc; ++i)
    {
        // Copy the arg
        char* arg = buffer;
        strcpy(arg, argv[i]);

        // Check if it's an option
        if (strlen(arg) > 1 && arg[0] == '-' && isalpha(arg[1]))
        {
            // Push previous option if necessary
            if (isOpt) opts.pushBack(opt);

            // Skip leading '-'s
            do
            {
                ++arg;
            } while (*arg == '-');

            // Look for an '='
            char* eq = strchr(arg, '=');
            if (eq)
            {
                *eq = 0;
                char* value = eq+1;

                opt.name = arg;
                opt.value = value;

                opts.pushBack(opt);
                isOpt = false;
            }
            else
            {
                opt.name = arg;
                opt.value = empty;

                isOpt = true;
            }
        }
        else
        {
            if (!isOpt) opt.name.clear();
            opt.value = arg;

            opts.pushBack(opt);
            isOpt = false;
        }
    }

    if (isOpt) opts.pushBack(opt);
}

void parseOptions(const std::string& filename, Array<Option>& opts)
{
    char arg[1024];
    Array<char*> argv;

    TextReader reader(makeRef<File>(filename));

    argv.pushBack(0);
    while (reader.readString(arg, sizeof(arg)) != EOF)
    {
        char* str = new char[strlen(arg)+1];
        strcpy(str, arg);
        argv.pushBack(str);
    }

    parseOptions(argv.getSize(), argv.getData(), opts);

    for (char* str : argv)
        delete[] str;
}

} // namespace prt
