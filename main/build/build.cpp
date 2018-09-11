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

#include "sys/common.h"
#include "sys/string.h"
#include "sys/sysinfo.h"
#include "sys/logging.h"
#include "sys/filesystem.h"
#include "sys/blob.h"
#include "sys/option.h"
#include "sys/props.h"
#include "sys/timer.h"
#include "geometry/triangle_mesh_builder.h"
#include "obj_loader.h"

namespace prt {

int mainBuild(int argc, char* argv[])
{
    std::cout << "ProtoRay Build" << std::endl;
	std::cout << std::endl;

	if (argc < 2)
	{
        std::cout << "Usage: protoray build [options]" << std::endl;
		return 0;
	}

    setLogFile("build.log");

    // Parse the options
    Array<Option> opts;
    parseOptions(argc, argv, opts);

    std::string type = "mesh";
    std::string input;
    std::string output;
    Props buildProps;

    for (Option& opt : opts)
    {
        if (opt.name.empty())
        {
            if (!input.empty())
            {
                LogError() << "Multiple inputs";
                return 1;
            }
            input = opt.value;
        }
        else if (opt.name == "o")
        {
            if (!output.empty())
            {
                LogError() << "Multiple outputs";
                return 1;
            }
            output = opt.value;
        }
        else if (opt.name == "t")
        {
            type = opt.value;
        }
        else
        {
            Log() << "Option: " << opt;
            buildProps.set(opt.name, opt.value);
        }
    }

    if (output.empty())
        output = getFilenameBase(input) + "." + type;

    std::string inputExt = getFilenameExt(input);

    // Load/build mesh
    TriangleMesh mesh;

    if (inputExt == "mesh")
    {
        // Load mesh
        loadBlob(input, mesh);
    }
    else
    {
        // Load soup
        TriangleSoup soup;

        if (inputExt == "obj")
        {
            ObjLoader objLoader;
            objLoader.load(input, soup);
        }
        else
        {
            LogError() << "Unsupported input format";
            return 1;
        }

        // Build mesh
        TriangleMeshBuilder(soup, mesh);
    }

    // Build
    if (type == "mesh")
    {
        // Mesh
        saveBlob(output, mesh);
    }
    else
    {
        LogError() << "Invalid type";
        return 1;
    }

	return 0;
}

} // namespace prt
