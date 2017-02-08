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

#pragma once

#include "sys/common.h"
#include "sys/memory.h"
#include "sys/string.h"
#include "sys/timer.h"
#include "math/vec3.h"
#include "image/surface.h"
#include "input.h"

namespace prt {

enum DisplayMode
{
    displayModeWindow,
    displayModeFullscreen,
    displayModeOffscreen,
};

class Window : Uncopyable
{
private:
    int width;
    int height;
    DisplayMode mode;

    // Text
    std::stringstream textBuffer;

    bool isRunning;
    bool isRefreshEnabled;
    bool isInputEnabled;
    Timer renderTimer;
    double renderTime;
    Timer displayTimer;
    double displayTime;
    std::string title;

public:
    Window(int width, int height, DisplayMode mode);
	virtual ~Window() {}

    // Starts the display loop
	bool run();

    int getWidth() const { return width; }
    int getHeight() const { return height; }
    DisplayMode getDisplayMode() const { return mode; }

    const std::string& getTitle() const { return title; }
    void setTitle(const std::string& str);

    bool getInputEnabled() const { return isInputEnabled; }
    void setInputEnabled(bool flag);

    bool getRefreshEnabled() const { return isRefreshEnabled; }
    void setRefreshEnabled(bool flag);

    double getRenderTime() const { return renderTime; }
    double getDisplayTime() const { return displayTime; }

protected:
	void quit();

	// Initialization called by run() at the beginning
    virtual void onInit() {}

	// Cleanup called by run() at the end
    virtual void onDestroy() {}

	// Rendering called by run() per frame
    // Call beginFrame() and endFrame()
    virtual void onRender() {}

    // Keyboard events (called before onRender)
    virtual void onKeyDown(int key) {}
    virtual void onKeyUp(int key) {}

    // Mouse events (called before onRender)
    virtual void onMouseButtonDown(int button, int x, int y) {}
    virtual void onMouseButtonUp(int button) {}
    virtual void onMouseMotion(int dx, int dy) {}

    // Functions callable inside onRender()
    void beginFrame(Surface& surface);
    void endFrame();

    // Text
    std::ostream& getText()
    {
        return textBuffer;
    }

private:
    bool initDisplay();
    bool initDisplayWindow();
    bool initDisplayOffscreen();
	void destroyDisplay();
	void handleInput();
};

} // namespace prt
