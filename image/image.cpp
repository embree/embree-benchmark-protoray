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

#include "sys/logging.h"
#include "image_io.h"
#include "image.h"

namespace prt {

bool loadImage(const std::string& filename, Image4uc& image)
{
    ImageIo::Handle handle = ImageIo::open(filename);
    if (!handle)
        throw std::runtime_error("could not load image: " + filename);

    ImageIo::Desc desc = ImageIo::getDesc(handle);
    if (desc.format != pixelFormatBgr8)
    {
        ImageIo::close(handle);
        throw std::runtime_error("pixel format mismatch");
    }

    image.alloc(desc.size);
    ImageIo::getData(handle, image.getData());
    ImageIo::close(handle);
}

bool loadImage(const std::string& filename, Image3f& image)
{
    ImageIo::Handle handle = ImageIo::open(filename);
    if (!handle)
        throw std::runtime_error("could not load image: " + filename);

    ImageIo::Desc desc = ImageIo::getDesc(handle);
    if (desc.format != pixelFormatRgb32f)
    {
        ImageIo::close(handle);
        throw std::runtime_error("pixel format mismatch");
    }

    image.alloc(desc.size);
    ImageIo::getData(handle, image.getData());
    ImageIo::close(handle);
}

bool saveImage(const std::string& filename, const Image4uc& image)
{
    if (filename.find(".") == std::string::npos)
        return saveImage(filename + ".ppm", image);

    ImageIo::Desc desc;
    desc.size = image.getSize();
    desc.format = pixelFormatBgr8;
    return ImageIo::save(filename, desc, image.getData());
}

bool saveImage(const std::string& filename, const Image<int>& image)
{
    if (filename.find(".png") == std::string::npos)
#ifdef IMAGE_SUPPORT
        return saveImage(filename + ".png", image);
#else
        return saveImage(filename + ".ppm", image);
#endif

#ifndef __MIC__
    // Host
    ImageIo::Desc desc;
    desc.size = image.getSize();
    desc.format = pixelFormatBgr8;
    return ImageIo::save(filename, desc, image.getData());
#else
    // Device
    throw std::runtime_error("saveImage not implemented");
#endif
}

bool saveImage(const std::string& filename, const Image3f& image)
{
    if (filename.find(".exr") == std::string::npos)
        return saveImage(filename + ".exr", image);

    ImageIo::Desc desc;
    desc.size = image.getSize();
    desc.format = pixelFormatRgb32f;
    return ImageIo::save(filename, desc, image.getData());
}

bool saveImage(const std::string& filename, const Image1f& image)
{
    if (filename.find(".exr") == std::string::npos)
        return saveImage(filename + ".exr", image);

    ImageIo::Desc desc;
    desc.size = image.getSize();
    desc.format = pixelFormatR32f;
    return ImageIo::save(filename, desc, image.getData());
}

} // namespace prt
