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

#include "text_reader.h"

namespace prt {

TextReader::TextReader(ref<Stream> sm)
{
    stream = sm;
    nextChar();
}

// Read the next character into the buffer
void TextReader::nextChar()
{
    char c;
    do
    {
        if (stream->read(&c, 1) == 0)
        {
            buffer = EOF;
            return;
        }
    } while (c == '\r');

    buffer = c;
}

int TextReader::readChar()
{
    int c = buffer;
    nextChar();
    return c;
}

int TextReader::peekChar()
{
    return buffer;
}

int TextReader::readLine(char* dest, int maxCount)
{
    assert(maxCount > 0);
    int count = 0;

    for (; ;)
    {
        int c = readChar();

        if (c == EOF && count == 0) return EOF;
        if (c == EOF || c == '\n') break;

        if (count < maxCount-1)
            dest[count++] = (char)c;
    }

    dest[count++] = 0;
    return count;
}

int TextReader::readString(char* dest, int maxCount)
{
    assert(maxCount > 0);
    int count = 0;

    for (; ;)
    {
        int c = peekChar();

        if (c == EOF)
        {
            if (count == 0) return EOF;
            break;
        }

        if (isspace(c))
        {
            if (count == 0)
            {
                readChar();
                continue;
            }
            break;
        }

        readChar();

        if (count < maxCount-1)
            dest[count++] = (char)c;
    }

    dest[count++] = 0;
    return count;
}

} // namespace prt
