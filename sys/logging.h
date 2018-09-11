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

#include <sstream>
#include "common.h"

namespace prt {

enum LogLevel
{
    logLevelInfo,
    logLevelWarn,
    logLevelError,
};

class LogMessage
{
private:
    std::ostringstream stream;
    LogLevel level;

public:
    explicit LogMessage(LogLevel level);
    ~LogMessage();

    template <class T>
    LogMessage& operator <<(const T& v)
    {
        stream << v;
        return *this;
    }
};


class Log : public LogMessage
{
public:
    Log() : LogMessage(logLevelInfo) {}
};

class LogWarn : public LogMessage
{
public:
    LogWarn() : LogMessage(logLevelWarn) {}
};

class LogError : public LogMessage
{
public:
    LogError() : LogMessage(logLevelError) {}
};


void initLogging();
void initLogging(const std::string& filename);
void setLogFile(const std::string& filename);
void setMinLogLevel(LogLevel level);
LogLevel getMinLogLevel();

} // namespace prt
