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

#include "sys/ref.h"
#include "sys/array.h"
#include "sys/props.h"
#include "sys/timer.h"
#include "render/device.h"
#include "window.h"
#include "view.h"
#include "stats_recorder.h"

namespace prt {

class RenderWindow : public Window
{
private:
    struct Tone
    {
        int typeId;
        float ev; // exposure value
        float burn;
    };

    ref<Device> device;

    View view;
    View prevView;
    View copiedView;
    int viewId;
    int defaultViewId;
    Vec3f viewPosDelta;
    float viewPosStep;
    float viewPosSpeed;
    float viewPosSpeedMul;
    float viewAngleStep;
    float viewFovDelta;
    float viewFovStep;
    float viewRadiusDelta;
    float viewRadiusStep;
    bool isMouseActive;
    bool isCameraRotateMode;
    ViewSet viewSet;
    std::string viewFilename;

    Tone tone;
    Tone prevTone;
    Array<std::string> toneMappers;

    Timer timer;
    double runTimeThreshold;  // sec
    Timer statsTimer;
    Timer autoShotTimer;
    double autoShotThreshold; // sec
    int frameCount;
    int spp;
    int maxSpp;

    Props frameStats;
    double currentFps;
    double currentMray;
    bool isFrameStatsPrinted;
    bool isTextEnabled;
    bool isBenchmarkMode;
    int warmupSpp;
    std::string resultPrefix;
    StatsRecorder statsRecorder;
    std::string deviceInfo;
    bool isDemoMode;
    bool printCamera;
    Props buildStats;

public:
    RenderWindow(int width, int height, DisplayMode mode, const ref<Device>& device, const Props& props, const Props& buildStats);

private:
    void onInit();
    void onDestroy();
    void onRender();

    void resetView();
    void setActiveView(int id);
    void saveView();

    void queryPixel(int x, int y);

    void onKeyDown(int key);
    void onKeyUp(int key);
    void onMouseButtonDown(int button, int x, int y);
    void onMouseButtonUp(int button);
    void onMouseMotion(int dx, int dy);

    void printFrameStats();
    void saveAvgStats(const Props& frameStats, const std::string& filename);
    void saveFullStats(const std::string& filename);
    void saveScreenshot(const std::string& filename);
    void saveAutoScreenshot();
};

} // namespace prt
