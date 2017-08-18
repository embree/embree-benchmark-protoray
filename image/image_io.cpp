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
#include "sys/filesystem.h"
#include "math/vec3.h"
#include "math/vec4.h"
#include "image_io.h"

namespace prt {

struct ImageIo::HandleSt
{
    ImageIo::Desc desc;
};

ImageIo::Handle ImageIo::open(const std::string& filename)
{
    LogError() << "Unsupported image file format";
    return 0;
}

ImageIo::Desc ImageIo::getDesc(Handle h)
{
    return h->desc;
}

void ImageIo::getData(Handle h, void* data)
{
}

void ImageIo::close(Handle h)
{
    delete h;
}

bool ImageIo::save(const std::string& filename, const ImageIo::Desc& desc, const void* data)
{
    Log() << "Saving image: " << filename;

    // Always use our own PPM writer
    if (desc.format == pixelFormatBgr8 && getFilenameExt(filename) == "ppm")
    {
        FILE* file = fopen(filename.c_str(), "wb");
        if (file == 0)
        {
            LogError() << "Could not save image";
            return false;
        }

        fprintf(file, "P6\n%d %d\n255\n", desc.size.x, desc.size.y);

        for (int y = 0; y < desc.size.y; ++y)
        {
            const Vec4uc* inLine = (const Vec4uc*)data + y * desc.size.x;

            for (int x = 0; x < desc.size.x; ++x)
            {
                fputc(inLine[x].x, file);
                fputc(inLine[x].y, file);
                fputc(inLine[x].z, file);
            }
        }

        fclose(file);
        return true;
    }

    // Only PPM is supported
    LogError() << "Unsupported image file format";
    return false;
}

} // namespace prt
