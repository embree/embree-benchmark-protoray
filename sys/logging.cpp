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

#include <iostream>
#include <iomanip>
#include <fstream>

#include <ctime>

#include "file.h"
#include "mutex.h"
#include "lock_guard.h"
#include "logging.h"

namespace prt {

namespace {

const char* logLevelNames[] = {"INFO", "WARNING", "ERROR"};

LogLevel minLogLevel = logLevelInfo;
bool isFileLoggingEnabled = false;
std::ofstream logFile;
Mutex logMutex;

// Always runs on the host
void addLogMessage(LogLevel level, const std::string& text, int devId = -1)
{
    if (level < minLogLevel)
        return;

    // Acquire log lock
    LockGuard<Mutex> lock(logMutex);

    // Get the current time
    time_t rawTime;
    time(&rawTime);
    tm* decodedTime = localtime(&rawTime);

    // Make the header
    std::stringstream header;

    header << std::setfill('0') << "[" <<
              std::setw(2) << decodedTime->tm_hour << ":" <<
              std::setw(2) << decodedTime->tm_min << ":" <<
              std::setw(2) << decodedTime->tm_sec <<
              "] ";

    if (level != logLevelInfo)
        header << "[" << logLevelNames[level] << "] ";

    if (devId >= 0)
        header << "[mic" << devId << "] ";

    // Send the message to stdout
    std::cout << header.str() << text << std::endl;

    // Save to file if necessary
    if (isFileLoggingEnabled)
    {
        // Make the full header
        std::stringstream fullHeader;

        fullHeader << std::setfill('0') << "[" <<
                      std::setw(4) << (decodedTime->tm_year + 1900) << "/" <<
                      std::setw(2) << (decodedTime->tm_mon + 1) << "/" <<
                      std::setw(2) << decodedTime->tm_mday <<
                      "|" <<
                      std::setw(2) << decodedTime->tm_hour << ":" <<
                      std::setw(2) << decodedTime->tm_min << ":" <<
                      std::setw(2) << decodedTime->tm_sec <<
                      "] ";

        if (level != logLevelInfo)
            fullHeader << "[" << logLevelNames[level] << "] ";

        if (devId >= 0)
            header << "[mic" << devId << "] ";

        // Save the message to the file
        logFile << fullHeader.str() << text << std::endl;
    }
}

} // namespace

void initLogging()
{
}

void initLogging(const std::string& filename)
{
    setLogFile(filename);
    initLogging();
}

void setLogFile(const std::string& filename)
{
    bool isExisting = File::exists(filename);
    logFile.open(filename.c_str(), std::ios::app);

    if (logFile.is_open())
    {
        isFileLoggingEnabled = true;
        if (isExisting)
            logFile << std::endl;
    }
    else
    {
        isFileLoggingEnabled = false;
        LogError() << "Could not open log file: " << filename;
    }
}

void setMinLogLevel(LogLevel level)
{
    minLogLevel = level;
}

LogLevel getMinLogLevel()
{
    return minLogLevel;
}

LogMessage::LogMessage(LogLevel level)
    : level(level)
{
    stream << std::fixed << std::setprecision(3);
}

LogMessage::~LogMessage()
{
    addLogMessage(level, stream.str());
}

} // namespace prt
