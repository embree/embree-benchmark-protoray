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

#include "sys/sysinfo.h"
#include "sys/logging.h"
#include "math/math.h"
#include "image/image.h"
#include "render_window.h"

namespace prt {

RenderWindow::RenderWindow(int width, int height, DisplayMode mode, const ref<Device>& device, const Props& props, const Props& buildStats)
    : Window(width, height, mode)
{
    this->device = device;

    resultPrefix = props.get("benchmark", "");
    isBenchmarkMode = !resultPrefix.empty();
    runTimeThreshold = props.get("runtime", double(posInf));
    autoShotThreshold = props.get("autoshot", 0.0);
    maxSpp = props.get("spp", 0);
    warmupSpp = props.get("warmup", maxSpp / 6);

    // Setup view
    viewFilename = props.get("viewFile", "default.view");
    defaultViewId = props.get("view", 0);

    prevView.pos = zero;
    prevView.angleX = 0;
    prevView.angleY = 0;
    prevView.fovY = -1.0f;

    printCamera = false;

    // Setup tone mapping
    toneMappers.pushBack("none", "linear", "reinhard");
    std::string toneMapper = props.get("tonemap", "none");
    tone.typeId = 0;
    for (int i = 0; i < toneMappers.getSize(); ++i)
    {
        if (toneMappers[i] == toneMapper)
        {
            tone.typeId = i;
            break;
        }
    }

    tone.ev = props.get("ev", 0.0f);
    tone.burn = props.get("burn", 0.5f);

    prevTone.typeId = -1;
    prevTone.ev = 0;
    prevTone.burn = 0;

    isTextEnabled = props.exists("overlay");
    isFrameStatsPrinted = false;
    currentFps = 0.0;
    currentMray = 0.0;

    deviceInfo = device->getInfo();
    isDemoMode = props.exists("demo");
    this->buildStats = buildStats;
}

void RenderWindow::onInit()
{
	// Reset all views
    viewId = -1;
	resetView();

    for (int i = 0; i < ViewSet::size; ++i)
        viewSet.views[i] = view;

	// Try to load the view set
    loadViewSet(viewFilename, viewSet);

    // Set the active view to the default
    setActiveView(defaultViewId);
    copiedView = view;

	// Compute camera movement parameters
    viewPosStep = reduceMin(device->getSceneBounds().getSize()) * 0.02f;
    viewPosSpeed = 1.0f;
    viewPosSpeedMul = 1.05f;
    viewAngleStep = 0.2f / (float)getHeight();
    viewPosDelta = zero;
    viewFovStep = degToRad(0.5f);
    viewFovDelta = 0.0f;
    viewRadiusStep = viewPosStep * 0.005f;
    viewRadiusDelta = 0.0f;

    isMouseActive = false;
    isCameraRotateMode = false;

    // Init the frame stats
    frameCount = 0;
    spp = 0;

    // Benchmark mode
    if (isBenchmarkMode)
	{
        Log() << "Benchmark mode";
        setInputEnabled(false);
	}

    timer.reset();
    statsTimer.reset();
    autoShotTimer.reset();
}

void RenderWindow::onDestroy()
{
    Props avgStats = buildStats;
    statsRecorder.getAverage(avgStats);
    Log() << "Average: " << avgStats;

    if (isBenchmarkMode)
    {
        saveAvgStats(avgStats, resultPrefix + ".txt");
        saveFullStats(resultPrefix + ".csv");
        saveScreenshot(resultPrefix);
    }
    else
    {
        if (getDisplayMode() == displayModeOffscreen)
            saveAutoScreenshot();
    }
}

void RenderWindow::onRender()
{   
	// Setup the camera
    Basis3f cameraBasis = Basis3f(one).rotateV(view.angleY).rotateU(view.angleX);
    view.pos += cameraBasis.toGlobal(viewPosDelta * viewPosStep * viewPosSpeed);
    view.fovY = clamp(view.fovY + viewFovDelta * viewFovStep, viewFovStep, degToRad(100.0f));
    view.radius = max(view.radius + viewRadiusDelta * viewRadiusStep * viewPosSpeed, 0.0f);

    // Update the camera if necessary
    bool isCameraUpdated = false;
    bool updateCamera = memcmp(&view, &prevView, sizeof(View)) != 0;
    if (updateCamera || printCamera)
    {
        Props camera;
        if (view.radius == 0.0f)
            camera.set("type", "pinhole");
        else
            camera.set("type", "thinlens");

        camera.set("position", view.pos);
        camera.set("basis", cameraBasis);
        camera.set("fov", view.fovY);
        camera.set("aspectRatio", (float)getWidth() / (float)getHeight());
        camera.set("lensRadius", view.radius);
        camera.set("focalDistance", view.focus);

        if (printCamera)
        {
            Log() << "Camera: " << std::setprecision(6) << camera;
            printCamera = false;
        }

        if (updateCamera)
        {
            device->initCamera(camera);
            prevView = view;
            isCameraUpdated = true;
        }
    }

    // Update tone mapping if necessary
    if (memcmp(&tone, &prevTone, sizeof(Tone)) != 0)
    {
        Props toneMapper;
        toneMapper.set("type", toneMappers[tone.typeId]);
        toneMapper.set("exposure", exp2(tone.ev));
        toneMapper.set("burn", tone.burn);

        device->initToneMapper(toneMapper);
        prevTone = tone;
    }

    // Clear the frame if necessary
    if (isCameraUpdated)
    {
        device->clearFrame();
        spp = 0;
    }

    // Render the frame
    bool isRendering = maxSpp == 0 || spp < maxSpp || isCameraUpdated;

    if (!isRendering && (getDisplayMode() == displayModeOffscreen || isBenchmarkMode))
        quit();

    if (isRendering)
    {
        // Init the stats
        frameStats.clear();
        frameStats.set("renderMs", 0.0);
        frameStats.set("fps", 0.0);

        isFrameStatsPrinted = false;
    }

    // Render the frame
    Surface surface;
    beginFrame(surface);

    if (isRendering)
        device->render(frameStats);

    if (surface.data)
        device->updateFrame(surface);

    if (isTextEnabled)
    {
        getText() << deviceInfo;
        if (isRendering)
        {
            getText() << std::endl << std::fixed << std::setprecision(1) << currentFps << " fps" << std::endl << currentMray << " Mray/s";
        }
    }

    endFrame();

    if (isRendering)
    {
        // Complete the stats
        ++frameCount;
        spp += frameStats.get("spp", 1);

        frameStats.set("renderMs", getRenderTime() * 1000.0);
        frameStats.set("fps", 1.0 / getDisplayTime());

        // Record the stats
        if (spp > warmupSpp || !isBenchmarkMode)
            statsRecorder.add(frameStats);

        // Print the stats if necessary
        frameStats.set("spp", spp);
        if (statsTimer.query() >= 1.0)
        {
            statsTimer.reset();
            printFrameStats();
        }
    }
    else
    {
        printFrameStats();
    }

    // Save a screenshot if necessary
    if (autoShotThreshold > 0)
    {
        if (autoShotTimer.query() >= autoShotThreshold)
        {
            autoShotTimer.reset();
            saveAutoScreenshot();
        }
    }

    if (timer.query() >= runTimeThreshold)
        quit();
}

void RenderWindow::printFrameStats()
{
    if (!isFrameStatsPrinted)
        Log() << frameStats;

    isFrameStatsPrinted = true;

    currentFps = frameStats.get("fps", 0.0);
    currentMray = frameStats.get("mray", 0.0);
}

void RenderWindow::saveAvgStats(const Props& stats, const std::string& filename)
{
    Log() << "Saving stats: " << filename;
    FILE* file = fopen(filename.c_str(), "wt");
    for (auto& i : stats)
        fprintf(file, "stats_%s=%s\n", i.first.c_str(), i.second.get<std::string>().c_str());
    fclose(file);
}

void RenderWindow::saveFullStats(const std::string& filename)
{
    Log() << "Saving stats: " << filename;
    statsRecorder.saveCsv(filename);
}

void RenderWindow::saveScreenshot(const std::string& filename)
{
    Vec2i imageSize(getWidth(), getHeight());

    printFrameStats();

    // LDR
    Image4uc image(imageSize);
    Surface surface;
    surface.width = imageSize.x;
    surface.height = imageSize.y;
    surface.pitch = imageSize.x * 4;
    surface.data = image.getData();
    device->updateFrame(surface);
    saveImage(filename, image);
}

void RenderWindow::saveAutoScreenshot()
{
	int index = 0;

	// Try to open the screenshot index file
	FILE* indexFile = fopen("screenshot_index", "rt");
	if (indexFile != 0)
	{
		fscanf(indexFile, "%d", &index);
		fclose(indexFile);
	}

	// Save the screenshot with the next index
    std::stringstream filenameBase;
    filenameBase << "screenshot_" << std::setfill('0') << std::setw(4) << index;
    saveScreenshot(filenameBase.str());

	// Save the index file
	++index;
	indexFile = fopen("screenshot_index", "wt");
	fprintf(indexFile, "%d", index);
	fclose(indexFile);
}

void RenderWindow::resetView()
{
    if (viewId >= 0)
        Log() << "Reset view: " << viewId;

    if (viewId > 0)
    {
        // Reset from view 0
        view = viewSet.views[0];
        return;
    }

	// Setup the camera
    Box3f sceneBox = device->getSceneBounds();
	Vec3f sceneCenter = sceneBox.getCenter();
    view.pos.x = sceneCenter.x;
    view.pos.y = sceneCenter.y;
    view.pos.z = 1.7f * reduceMax(sceneBox.getSize()) + sceneCenter.z;

    view.angleX = 0.0f;
    view.angleY = 0.0f;

    view.fovY = degToRad(45.0f);

    view.radius = 0.0f;
    view.focus = length(sceneCenter - view.pos);
}

void RenderWindow::setActiveView(int id)
{
    viewId = id;
    view = viewSet.views[id];

    Log() << "Active view: " << viewId;
}

void RenderWindow::saveView()
{
    viewSet.views[viewId] = view;
    saveViewSet(viewFilename, viewSet);

    Log() << "Saved view: " << viewId;
}

void RenderWindow::queryPixel(int x, int y)
{
    Props result;
    result = device->queryPixel(x, y);
    if (result.isEmpty()) return;

    Log() << "Pixel: " << result;

    // Refocus
    if (view.radius > 0.0f)
        view.focus = result.get("depth", view.focus);
}

void RenderWindow::onKeyDown(int key)
{
	switch (key)
	{
    case keyUp:
	case 'w':
        viewPosDelta.z = -1.0f;
		break;

    case keyDown:
	case 's':
        viewPosDelta.z = 1.0f;
		break;

    case keyLeft:
	case 'a':
        viewPosDelta.x = -1.0f;
		break;

    case keyRight:
	case 'd':
        viewPosDelta.x = 1.0f;
		break;

    case keyNumpadMinus:
    case 'q':
        viewFovDelta = 1.0f;
		break;

    case keyNumpadPlus:
    case 'e':
        viewFovDelta = -1.0f;
		break;

    case '-':
        viewRadiusDelta = -1.0f;
        break;

    case '=':
        viewRadiusDelta = 1.0f;
        break;

    case 'p':
        view.radius = 0.0f;
        Log() << "Pinhole lens";
        break;

	case ' ':
		saveAutoScreenshot();
		break;

    case keyEscape:
		quit();
		break;

	default:
		// Change active view
		if (key >= '0' && key <= '9')
			setActiveView(key - '0');
		break;
	}

    if (isDemoMode)
        return;

    // In non-demo mode only
    switch (key)
    {
    // Reset view
    case keyBackspace:
        resetView();
        break;

    // Save view
    case keyReturn:
        saveView();
        break;

    // Copy view
    case 'c':
        copiedView = view;
        Log() << "Copied view";
        break;

    // Paste view
    case 'v':
        view = copiedView;
        Log() << "Pasted view";
        break;

    // Print camera
    case 'i':
        printCamera = true;
        break;

    // Tone mapping
    case 't':
        tone.typeId = (tone.typeId+1) % toneMappers.getSize();
        Log() << "Tone mapping: " << toneMappers[tone.typeId];
        break;

    case '[':
        tone.ev -= 0.1f;
        Log() << "EV: " << tone.ev;
        break;

    case ']':
        tone.ev += 0.1f;
        Log() << "EV: " << tone.ev;
        break;

    case ',':
        tone.burn = clamp(tone.burn - 0.02f, 0.0f, 1.0f);
        Log() << "Burn: " << tone.burn;
        break;

    case '.':
        tone.burn = clamp(tone.burn + 0.02f, 0.0f, 1.0f);
        Log() << "Burn: " << tone.burn;
        break;

    // Toggle refresh
    case 'r':
        setRefreshEnabled(!getRefreshEnabled());
        break;

    case 'o':
        isTextEnabled = !isTextEnabled;
        break;
    }
}

void RenderWindow::onKeyUp(int key)
{
	switch (key)
	{
    case keyUp:
	case 'w':
    case keyDown:
    case 's':
        viewPosDelta.z = 0.0f;
		break;

    case keyLeft:
	case 'a':
    case keyRight:
    case 'd':
        viewPosDelta.x = 0.0f;
		break;

    case keyNumpadMinus:
    case 'q':
    case keyNumpadPlus:
    case 'e':
        viewFovDelta = 0.0f;
		break;

    case '-':
    case '=':
        viewRadiusDelta = 0.0f;
        break;
	}
}

void RenderWindow::onMouseButtonDown(int button, int x, int y)
{
	switch (button)
	{
    case mouseButtonLeft:
        isMouseActive = true;
        isCameraRotateMode = true;
		break;

    case mouseButtonRight:
        queryPixel(x, y);

    case mouseButtonWheelUp:
        viewPosSpeed = min(viewPosSpeed*viewPosSpeedMul, 1e10f);
		break;

    case mouseButtonWheelDown:
        viewPosSpeed /= viewPosSpeedMul;
		break;
	}
}

void RenderWindow::onMouseButtonUp(int button)
{
	switch (button)
	{
    case mouseButtonLeft:
        isMouseActive = false;
		break;
	}
}

void RenderWindow::onMouseMotion(int dx, int dy)
{
    if (!isMouseActive)
		return;

    if (isCameraRotateMode)
    {
        view.angleX += dy * viewAngleStep;
        view.angleY += dx * viewAngleStep;

        if (view.angleX > float(pi)/2.0f)
            view.angleX = float(pi)/2.0f;
        else if (view.angleX < -float(pi)/2.0f)
            view.angleX = -float(pi)/2.0f;
    }
}

} // namespace prt
