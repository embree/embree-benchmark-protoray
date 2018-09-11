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

#include <csignal>
#include "sys/constants.h"
#include "sys/logging.h"
#include "window.h"

namespace prt {

// Interrupt signal handler
namespace
{
	volatile sig_atomic_t isInterrupted = 0;

	void handleInterrupt(int signal)
	{
		isInterrupted = 1;
        Log() << "Interrupted";
	}
}

Window::Window(int width, int height, DisplayMode mode)
{
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("resolution is invalid");

    this->width = width;
    this->height = height;
    this->mode = mode;
    isRunning = false;
    isRefreshEnabled = true;
    isInputEnabled = mode != displayModeOffscreen;
}

void Window::setTitle(const std::string& str)
{
    title = str;
}

bool Window::run()
{
    if (!initDisplay())
		return false;

    onInit();

	// Set interrupt signal handler
	isInterrupted = 0;
	signal(SIGINT, handleInterrupt);

    isRunning = true;
    Log() << "Render loop";

    // Start measuring display time
    renderTime = 0.0f;
    displayTime = posInf;
    displayTimer.reset();

    while (isRunning)
	{
		// Handle keyboard and mouse input
		handleInput();

		// Render the frame
        renderTime = 0.0f;
        displayTime = posInf;
        onRender();
	}

    onDestroy();
	destroyDisplay();

	// Restore signal handler
	signal(SIGINT, SIG_DFL);

	return true;
}

void Window::beginFrame(Surface& surface)
{
    surface.data = 0;

    // Start measuring render time
    renderTimer.reset();
}

void Window::endFrame()
{
    // Stop measuring render time
    renderTime = renderTimer.query();

    // Clear the text buffer
    textBuffer.str(std::string());

    // Stop measuring display time
    displayTime = displayTimer.query();
    displayTimer.reset();
}

void Window::quit()
{
    isRunning = false;
}

bool Window::initDisplay()
{
    Log() << "Resolution: " << width << "x" << height;

    if (mode == displayModeOffscreen)
        return initDisplayOffscreen();

    return initDisplayWindow();
}

bool Window::initDisplayWindow()
{
    LogError() << "GUI not supported";
    return false;
}

// Initialize an offscreen framebuffer
bool Window::initDisplayOffscreen()
{
	Log() << "Offscreen mode";

	return true;
}

void Window::destroyDisplay()
{
}

void Window::handleInput()
{
	if (isInterrupted)
	{
		quit();
		return;
	}
}

void Window::setInputEnabled(bool flag)
{
    isInputEnabled = flag;

    if (isInputEnabled)
		Log() << "Input: enabled";
	else
		Log() << "Input: disabled";
}

void Window::setRefreshEnabled(bool flag)
{
    isRefreshEnabled = flag;

    if (isRefreshEnabled)
        Log() << "Refresh: enabled";
	else
        Log() << "Refresh: disabled";
}

} // namespace prt
