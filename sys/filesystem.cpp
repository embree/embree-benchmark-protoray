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

#include <fstream>
#include "math/random.h"
#include "mutex.h"
#include "lock_guard.h"
#include "filesystem.h"

namespace prt {

std::string getFilenameBase(const std::string& filename)
{
    // FIXME
    return filename.substr(0, filename.find_last_of('.'));
}

std::string getFilenameExt(const std::string& filename)
{
    // FIXME
    return filename.substr(filename.find_last_of('.') + 1);
}

std::string convertFilename(const std::string& filename)
{
    std::string newFilename = filename;
    std::replace(newFilename.begin(), newFilename.end(), '\\', '/');
    return newFilename;
}

// Temp files
// ----------

namespace
{
    std::string tempPath;
    Random random(generateRandomSeed());
    Mutex mutex;
}

std::string getTempPath()
{
    return tempPath;
}

void setTempPath(const std::string& path)
{
    std::string newTempPath = convertFilename(path);

    // Remove trailing slashes
    for (int i = static_cast<int>(newTempPath.size() - 1); i >= 0; --i)
    {
        if (newTempPath[i] != '/')
        {
            newTempPath.erase(i + 1);
            break;
        }
    }

    tempPath = newTempPath + '/';
}

std::string makeTempFilename()
{
    LockGuard<Mutex> lockGuard(mutex);

	std::stringstream stringStream;
	stringStream << tempPath << ".temp-";

	for (int i = 0; i < 4; ++i)
	{
		stringStream << std::hex << std::setw(8) << std::setfill('0') << random.getUint();
	}

	return stringStream.str();
}

} // namespace prt
