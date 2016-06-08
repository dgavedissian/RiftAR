#include "Common.h"

#include "lib/Rectangle2D.h"
#include "lib/Entity.h"
#include "lib/Shader.h"

#include "RiftOutput.h"
#include "DebugOutput.h"

#include "KFusionTracker.h"

#include "RiftAR.h"

#define RIFT_DISPLAY
#define ENABLE_ZED
//#define DEBUG_FIND_OBJECT

DEFINE_MAIN(RiftAR);

float focalDepth = 1.5f;

RiftAR::RiftAR() :
    mAddArtificalLatency(false)
{
}

void RiftAR::init()
{
    // Set up the cameras
#ifdef ENABLE_ZED
    mZed = new ZEDCamera(sl::zed::HD720, 60);
#endif
    mRealsense = new RealsenseCamera(640, 480, 60, RealsenseCamera::ENABLE_COLOUR | RealsenseCamera::ENABLE_DEPTH);

    // Grab parameters from the destination camera
    cv::Size destinationSize;
#ifdef ENABLE_ZED
    bool invertColours = true;
    float fovh = mZed->getIntrinsics(ZEDCamera::LEFT).fovH;
    destinationSize.width = mZed->getWidth(ZEDCamera::LEFT);
    destinationSize.height = mZed->getHeight(ZEDCamera::LEFT);
#else
    bool invertColours = false;
    float fovh = mRealsense->getIntrinsics(RealsenseCamera::COLOUR).fovH;
    destinationSize.width = mRealsense->getWidth(RealsenseCamera::COLOUR);
    destinationSize.height = mRealsense->getHeight(RealsenseCamera::COLOUR);
#endif

    // Set up scene
    mRenderCtx = new Renderer(invertColours, 0.01f, 10.0f, USHRT_MAX * mRealsense->getDepthScale());
    mRenderCtx->lookingForHead = false;
    mRenderCtx->foundTransform = false;
    mRenderCtx->backbufferSize = getSize();

    // Initialise tracking system
    mTracking = new KFusionTracker(mRealsense);

    // Set up depth warping
    setupDepthWarpStream(destinationSize);

    // Set up output
#ifdef ENABLE_ZED
    mRenderCtx->colourTextures[0] = mZed->getTexture(ZEDCamera::LEFT);
    mRenderCtx->colourTextures[1] = mZed->getTexture(ZEDCamera::RIGHT);
    mRenderCtx->projection = mZed->getIntrinsics(ZEDCamera::LEFT).buildGLProjection(mRenderCtx->mZNear, mRenderCtx->mZFar);
#else
    mRenderCtx->colourTextures[0] = mRealsense->getTexture(RealsenseCamera::COLOUR);
    mRenderCtx->colourTextures[1] = mRealsense->getTexture(RealsenseCamera::COLOUR);
    mRenderCtx->projection = mRealsense->getIntrinsics(RealsenseCamera::COLOUR).buildGLProjection(mRenderCtx->mZNear, mRenderCtx->mZFar);
#endif

#ifdef RIFT_DISPLAY
    mOutputCtx = new RiftOutput(getSize(), destinationSize.width, destinationSize.height, fovh, invertColours);
#else
    mOutputCtx = new DebugOutput();
#endif

    // Enable culling, depth testing and stencil testing
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);

    // Capture a single frame and fork to the capture thread
#ifdef ENABLE_ZED
    mZed->capture();
    mZed->copyData();
#endif
    mRealsense->capture();
    mRealsense->copyData();
    mHasFrame = true;
    mIsCapturing = true;
    mCaptureThread = std::thread(&RiftAR::captureLoop, this);
}

RiftAR::~RiftAR()
{
    mIsCapturing = false;
    mCaptureThread.join();

    delete mRenderCtx;

    delete mTracking;

#ifdef ENABLE_ZED
    delete mZed;
#endif
    delete mRealsense;

    delete mOutputCtx;
}

#include <tuple>

void findCurves(const std::vector<int>& histogram, std::vector<std::tuple<int, int, int>>& curves)
{
    bool inCurve = false;
    int start;
    int size;
    for (int i = 0; i < histogram.size(); i++)
    {
        // Attempt to find the start of a curve
        if (!inCurve)
        {
            if (histogram[i] > 0)
            {
                inCurve = true;
                start = i;
                size = histogram[i];
            }
        }
        else // Attempt to find the end of the curve
        {
            size += histogram[i];
            if (i == (histogram.size() - 1) || histogram[i + 1] == 0) // short circuiting will prevent array out of bounds
            {
                inCurve = false;
                curves.push_back(std::make_tuple(start, i, size));
            }
        }
    }
}

bool findObject(cv::Mat image, float depthScale, float& objectDistance)
{
    // Set up a histogram
    std::vector<int> counts;
    const int CLASS_COUNT = 20;
    counts.resize(CLASS_COUNT);

    // Map pixel to class number
    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            uint16_t rawDepth = image.at<uint16_t>(r, c);

            // Ignore indeterminate values
            if (rawDepth == 0)
                continue;

            // Convert raw depth into 0-1 range
            float rawDepthScaled = (float)rawDepth / (float)0xffff;
            counts[(int)(rawDepthScaled * CLASS_COUNT)]++;
        }
    }

#ifdef DEBUG_FIND_OBJECT
    // Display histogram for debugging
    for (int c : counts)
        cout << c << ", " << endl;
    cout << endl;
#endif

    // Extract curves in form min, max (inclusive), area
    std::vector<std::tuple<int, int, int>> curves;
    findCurves(counts, curves);

    // If there are no curves, then there are no depth values
    if (curves.empty())
        return false;

    // Find the largest curve by area
    auto largestCurve = curves.begin();
    for (auto it = curves.begin(); it != curves.end(); it++)
    {
        if (std::get<2>(*it) > std::get<2>(*largestCurve))
            largestCurve = it;
    }

    // Find average of all points between the start and end of the first curve
    float sum = 0.0f;
    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            uint16_t rawDepth = image.at<uint16_t>(r, c);

            // Ignore indeterminate values
            if (rawDepth == 0)
                continue;

            // Convert raw depth into class
            float rawDepthScaled = (float)rawDepth / (float)0xffff;
            int depthClass = (int)(rawDepthScaled * CLASS_COUNT);

            // If class is within range defined above, consider it
            if (depthClass >= std::get<0>(*largestCurve) && depthClass <= std::get<1>(*largestCurve))
                sum += rawDepth * depthScale;
        }
    }

    // Return average
    objectDistance = sum / std::get<2>(*largestCurve);
    return true;
}

void RiftAR::render()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Attempt to grab the next frame
    getFrame();

    // Add artificial latency
    if (mAddArtificalLatency)
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Update the cameras pose
    mTracking->update(mDepthFrame);
    mRenderCtx->view = glm::inverse(mTracking->getCameraPose());

    // Search for the head
    if (mTracking->checkTargetPosition(mRenderCtx->headTransform))
    {
        mRenderCtx->lookingForHead = false;
        mRenderCtx->foundTransform = true;
        mRenderCtx->alignmentEntity->setTransform(mRenderCtx->headTransform);
        mRenderCtx->expandedAlignmentEntity->setTransform(mRenderCtx->headTransform * glm::scale(glm::mat4(), glm::vec3(1.1f, 1.1f, 1.1f)));
        mRenderCtx->overlay->setTransform(mRenderCtx->headTransform);
    }

    // Warp depth textures for occlusion
    mRealsenseDepth->warpToPair(mDepthFrame, mZedCalib, mRenderCtx->eyeMatrix[0], mRenderCtx->eyeMatrix[1]);

    // Find centre object
    cv::Size2i regionSize(64, 64);
    cv::Rect region((mDepthFrame.cols - regionSize.width) / 2, (mDepthFrame.rows - regionSize.height) / 2, regionSize.width, regionSize.height);
    float hit = 0.0f;
    bool found = findObject(mDepthFrame(region), mRealsense->getDepthScale(), focalDepth);
    if (found)
    {
#ifdef ENABLE_ZED
        // Calculate HIT based on focal depth
        float baseline = mZed->getBaseline();
        float convergence = mZed->getConvergence();
        float focalPoint = baseline / (2.0f * tan(convergence * 0.5f));
        float d = baseline * (1.0f - focalPoint / focalDepth);
        CameraIntrinsics& intr = mZed->getIntrinsics(ZEDCamera::LEFT);
        hit = intr.cameraMatrix.at<double>(0, 0) * d / focalPoint;
        cout << focalPoint << "/" << focalDepth << " - " << d << " - " << hit << endl;
#endif
    }

    // Render scene
    mOutputCtx->renderScene(mRenderCtx, (int)hit);
}

void RiftAR::keyEvent(int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_SPACE)
        {
            mRenderCtx->toggleDebug();
        }

        if (key == GLFW_KEY_S)
        {
            mTracking->beginSearchingFor(mRenderCtx->alignmentEntity);
            mRenderCtx->lookingForHead = true;
        }

        if (key == GLFW_KEY_R)
        {
            mTracking->reset();
        }

        if (key == GLFW_KEY_L)
        {
            mAddArtificalLatency = !mAddArtificalLatency;
        }
    }
}

void RiftAR::scrollEvent(double x, double y)
{
}

cv::Size RiftAR::getSize()
{
    return cv::Size(1600, 600);
}

// Distortion
void RiftAR::setupDepthWarpStream(cv::Size destinationSize)
{
    mRealsenseDepth = new RealsenseDepthAdjuster(mRealsense, destinationSize);
    mRenderCtx->depthTextures[0] = mRealsenseDepth->getDepthTexture(0);
    mRenderCtx->depthTextures[1] = mRealsenseDepth->getDepthTexture(1);

    // Read parameters
#ifdef ENABLE_ZED
    mZedCalib = convertCVToMat3<double>(mZed->getIntrinsics(ZEDCamera::LEFT).cameraMatrix);
#else
    mZedCalib = convertCVToMat3<double>(mRealsense->getIntrinsics(RealsenseCamera::COLOUR).cameraMatrix);
#endif

    // Read extrinsics parameters that map the ZED to the realsense colour camera, and invert
    // to map in the opposite direction
#ifdef ENABLE_ZED
    cv::FileStorage fs("../stereo-params.xml", cv::FileStorage::READ);
    cv::Mat rotationMatrix, translation;
    fs["R"] >> rotationMatrix;
    fs["T"] >> translation;
    glm::mat4 realsenseColourToZedLeft = glm::inverse(buildExtrinsic(
        convertCVToMat3<double>(rotationMatrix),
        convertCVToVec3<double>(translation)));
#else
    glm::mat4 realsenseColourToZedLeft; // identity
#endif

    // Extrinsics to map from depth to colour in the F200
    glm::mat4 depthToColour = mRealsense->getExtrinsics(RealsenseCamera::DEPTH, RealsenseCamera::COLOUR);

    // Combined extrinsics mapping realsense depth to ZED left
    mRealsenseToZedLeft = realsenseColourToZedLeft * depthToColour;

#ifdef ENABLE_ZED
    mRenderCtx->eyeMatrix[0] = mRealsenseToZedLeft;
    mRenderCtx->eyeMatrix[1] = mZed->getExtrinsics(ZEDCamera::LEFT, ZEDCamera::RIGHT) * mRealsenseToZedLeft;
#else
    mRenderCtx->eyeMatrix[0] = mRenderCtx->eyeMatrix[1] = mRealsenseToZedLeft;
#endif
}

void RiftAR::captureLoop()
{
    while (mIsCapturing)
    {
        // Get current pose
#ifdef RIFT_DISPLAY
        int frameIndex = mFrameIndex;
        ovrPosef eyePose[2];
        static_cast<RiftOutput*>(mOutputCtx)->newFrame(frameIndex, eyePose);
#endif

        // Capture from the cameras
#ifdef ENABLE_ZED
        mZed->capture();
#endif
        mRealsense->capture();

        // Copy captured data and pose information
        std::lock_guard<std::mutex> guard(mFrameMutex);
#ifdef RIFT_DISPLAY
        mFrameIndex = frameIndex;
        mEyePose[0] = eyePose[0];
        mEyePose[1] = eyePose[1];
#endif
#ifdef ENABLE_ZED
        mZed->copyData();
#endif
        mRealsense->copyData();

        // We now have a frame
        mHasFrame = true;
    }
}

bool RiftAR::getFrame()
{
    if (!mHasFrame)
        return false;

    std::lock_guard<std::mutex> guard(mFrameMutex);

    // Set the current frame pose
#ifdef RIFT_DISPLAY
    static_cast<RiftOutput*>(mOutputCtx)->setFramePose(mFrameIndex, mEyePose);
#endif

    // Copy frames from the cameras
    mRealsense->copyFrameIntoCVImage(RealsenseCamera::DEPTH, &mDepthFrame);
    mRealsense->updateTextures();
#ifdef ENABLE_ZED
    mZed->updateTextures();
#endif

    // Mark this frame as consumed
    mHasFrame = false;

    return true;
}
