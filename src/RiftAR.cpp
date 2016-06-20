#include "Common.h"

#include "lib/Rectangle2D.h"
#include "lib/Entity.h"
#include "lib/Shader.h"
#include "lib/Timer.h"
#include "lib/TextureCV.h"

#include "RiftOutput.h"
#include "DebugOutput.h"

#include "FindObject.h"
#include "KFusionTracker.h"

#include "RiftAR.h"

//#define RIFT_DISPLAY
//#define ENABLE_ZED

DEFINE_MAIN(RiftAR);

RiftAR::RiftAR() :
    mAddArtificalLatency(false)
{
}

void RiftAR::init()
{
    // Set up the cameras
#ifdef ENABLE_ZED
    mZed = make_unique<ZEDCamera>(sl::zed::HD720, 60);
#endif
    mRealsense = make_unique<RealsenseCamera>(640, 480, 60, RealsenseCamera::ENABLE_COLOUR | RealsenseCamera::ENABLE_DEPTH);

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

    // Initialise tracking system
    mTracking = make_unique<KFusionTracker>(mRealsense.get());

    // Set up scene
    glm::mat4 projection;
    float znear = 0.01f, zfar = 10.0f;
#ifdef ENABLE_ZED
    projection = mZed->getIntrinsics(ZEDCamera::LEFT).buildGLProjection(znear, zfar);
#else
    projection = mRealsense->getIntrinsics(RealsenseCamera::COLOUR).buildGLProjection(znear, zfar);
#endif
    mRenderer = make_unique<Renderer>(projection, znear, zfar, getSize(), USHRT_MAX * mRealsense->getDepthScale(), mTracking.get(), invertColours);

    // Set up output
    GLuint colourTextures[2];
#ifdef ENABLE_ZED
    colourTextures[0] = mZed->getTexture(ZEDCamera::LEFT);
    colourTextures[1] = mZed->getTexture(ZEDCamera::RIGHT);
#else
    colourTextures[0] = mRealsense->getTexture(RealsenseCamera::COLOUR);
    colourTextures[1] = mRealsense->getTexture(RealsenseCamera::COLOUR);
#endif

    // Set up depth warping
    mRealsenseDepth = make_unique<RealsenseDepthAdjuster>(mRealsense.get(), destinationSize);
    GLuint depthTextures[2];
    depthTextures[0] = mRealsenseDepth->getDepthTexture(0);
    depthTextures[1] = mRealsenseDepth->getDepthTexture(1);

    // Grab the displayed cameras intrinsics
#ifdef ENABLE_ZED
    mDisplayIntr = convertCVToMat3<double>(mZed->getIntrinsics(ZEDCamera::LEFT).cameraMatrix);
#else
    mDisplayIntr = convertCVToMat3<double>(mRealsense->getIntrinsics(RealsenseCamera::COLOUR).cameraMatrix);
#endif

    // Read extrinsics parameters that map the realsense colour camera to the ZED (if enabled)
    glm::mat4 rsColourToZedLeft;
#ifdef ENABLE_ZED
    cv::FileStorage fs("../stereo-params.xml", cv::FileStorage::READ);
    cv::Mat rotationMatrix, translation;
    fs["R"] >> rotationMatrix;
    fs["T"] >> translation;
    rsColourToZedLeft = buildExtrinsic(
        convertCVToMat3<double>(rotationMatrix),
        convertCVToVec3<double>(translation));
#endif

    // Extrinsics to map from depth to left and right eye
    glm::mat4 rsDepthToColour = mRealsense->getExtrinsics(RealsenseCamera::DEPTH, RealsenseCamera::COLOUR);
    glm::mat4 rsDepthToZedLeft = rsColourToZedLeft * rsDepthToColour;
#ifdef ENABLE_ZED
    mExtrRsToZed[0] = rsDepthToZedLeft;
    mExtrRsToZed[1] = mZed->getExtrinsics(ZEDCamera::LEFT, ZEDCamera::RIGHT) * rsDepthToZedLeft;
#else
    mExtrRsToZed[0] = mExtrRsToZed[1] = rsDepthToZedLeft;
#endif

    // Inform the renderer of our camera settings
    mRenderer->setTextures(colourTextures, depthTextures);
    mRenderer->setExtrinsics(mExtrRsToZed);
    
    // Set up output
#ifdef RIFT_DISPLAY
    mOutputCtx = make_unique<RiftOutput>(getSize(), destinationSize.width, destinationSize.height, fovh, invertColours);
#else
    mOutputCtx = make_unique<DebugOutput>();
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
    mRenderer->setViewMatrix(glm::inverse(mTracking->getCameraPose()));

    // Search for the head
    glm::mat4 transform;
    if (mTracking->checkTargetPosition(transform))
        mRenderer->setObjectFound(transform);

    // Warp depth textures for occlusion
    mRealsenseDepth->warpToPair(mDepthFrame, mDisplayIntr, mExtrRsToZed[0], mExtrRsToZed[1]);

    // Find centre object
    cv::Size2i regionSize(64, 64);
    cv::Rect region((mDepthFrame.cols - regionSize.width) / 2, (mDepthFrame.rows - regionSize.height) / 2, regionSize.width, regionSize.height);
    int hit = 0;
    float focalDepth;
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
    mOutputCtx->renderScene(mRenderer.get(), hit);
}

void RiftAR::keyEvent(int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_Q)
        {
            mRenderer->setState(RS_COLOUR);
        }

        if (key == GLFW_KEY_W)
        {
            mRenderer->setState(RS_DEBUG_DEPTH);
        }

        if (key == GLFW_KEY_E)
        {
            mRenderer->setState(RS_DEBUG_KFUSION);
        }

        if (key == GLFW_KEY_SPACE)
        {
            mRenderer->beginSearchingFor(Entity::loadModel("../media/meshes/bob-smooth.stl"));
            mTracking->beginSearchingFor(mRenderer->getTargetEntity());
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

void RiftAR::captureLoop()
{
    while (mIsCapturing)
    {
        // Get current pose and increment the frame index
#ifdef RIFT_DISPLAY
        RiftPose poseState = mPoseState;
        static_cast<RiftOutput*>(mOutputCtx.get())->newFrame(poseState.frameIndex, poseState.eyePose);
#endif

        // Capture from the cameras
#ifdef ENABLE_ZED
        mZed->capture();
#endif
        mRealsense->capture();

        // Copy captured data and pose information
        std::lock_guard<std::mutex> guard(mFrameMutex);
#ifdef RIFT_DISPLAY
        mPoseState = poseState;
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
    static_cast<RiftOutput*>(mOutputCtx.get())->setFramePose(mPoseState.frameIndex, mPoseState.eyePose);
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
