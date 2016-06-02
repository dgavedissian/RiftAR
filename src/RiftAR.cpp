#include "Common.h"
#include "RiftAR.h"

#include "RiftOutput.h"
#include "DebugOutput.h"

#include "KFusionTracker.h"

//#define RIFT_DISPLAY
//#define ENABLE_ZED

DEFINE_MAIN(RiftAR);

RiftAR::RiftAR() :
    mAddArtificalLatency(false)
{
}

void RiftAR::init()
{
    mRenderCtx.alignmentModel = nullptr;
    mRenderCtx.model = nullptr;

    // Set up the cameras
#ifdef ENABLE_ZED
    mZed = new ZEDCamera(sl::zed::HD720, 60);
#endif
    mRealsense = new RealsenseCamera(640, 480, 60, RealsenseCamera::ENABLE_COLOUR | RealsenseCamera::ENABLE_DEPTH);

    // Get the width/height of the output colour stream that the user sees
    cv::Size destinationSize;
#ifdef ENABLE_ZED
    destinationSize.width = mZed->getWidth(ZEDCamera::LEFT);
    destinationSize.height = mZed->getHeight(ZEDCamera::LEFT);
#else
    destinationSize.width = mRealsense->getWidth(RealsenseCamera::COLOUR);
    destinationSize.height = mRealsense->getHeight(RealsenseCamera::COLOUR);
#endif

    // Initialise tracking system
    mTracking = new KFusionTracker(mRealsense);

    // Set up depth warping
    setupDepthWarpStream(destinationSize);

    // Set up scene
    mRenderCtx.lookingForHead = false;
    mRenderCtx.foundTransform = false;
    mRenderCtx.backbufferSize = getSize();
    mRenderCtx.depthScale = USHRT_MAX * mRealsense->getDepthScale();
    mRenderCtx.znear = 0.01f;
    mRenderCtx.zfar = 10.0f;
    mRenderCtx.alignmentModel = new Model("../media/meshes/bob-smooth.stl");
    mRenderCtx.expandedAlignmentModel = new Model("../media/meshes/bob-smooth.stl");
    mRenderCtx.model = new Model("../media/meshes/graymatter.stl");

    // Set up output
#ifdef ENABLE_ZED
    bool invertColours = true;
    float width = mZed->getWidth(ZEDCamera::LEFT);
    float height = mZed->getHeight(ZEDCamera::LEFT);
    float fovh = mZed->getIntrinsics(ZEDCamera::LEFT).fovH;
    mRenderCtx.colourTextures[0] = mZed->getTexture(ZEDCamera::LEFT);
    mRenderCtx.colourTextures[1] = mZed->getTexture(ZEDCamera::RIGHT);
    mRenderCtx.projection = mZed->getIntrinsics(ZEDCamera::LEFT).buildGLProjection(mRenderCtx.znear, mRenderCtx.zfar);
#else
    bool invertColours = false;
    float width = mRealsense->getWidth(RealsenseCamera::COLOUR);
    float height = mRealsense->getHeight(RealsenseCamera::COLOUR);
    float fovh = mRealsense->getIntrinsics(RealsenseCamera::COLOUR).fovH;
    mRenderCtx.colourTextures[0] = mRealsense->getTexture(RealsenseCamera::COLOUR);
    mRenderCtx.colourTextures[1] = mRealsense->getTexture(RealsenseCamera::COLOUR);
    mRenderCtx.projection = mRealsense->getIntrinsics(RealsenseCamera::COLOUR).buildGLProjection(mRenderCtx.znear, mRenderCtx.zfar);
#endif

#ifdef RIFT_DISPLAY
    mOutputCtx = new RiftOutput(getSize(), width, height, fovh, invertColours);
#else
    mOutputCtx = new DebugOutput(mRenderCtx, invertColours);
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

    if (mRenderCtx.alignmentModel)
        delete mRenderCtx.alignmentModel;
    if (mRenderCtx.model)
        delete mRenderCtx.model;

    delete mTracking;

#ifdef ENABLE_ZED
    delete mZed;
#endif
    delete mRealsense;

    delete mOutputCtx;
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
    mRenderCtx.view = glm::inverse(mTracking->getCameraPose());

    // Search for the head
    if (mTracking->checkTargetPosition(mRenderCtx.headTransform))
    {
        mRenderCtx.lookingForHead = false;
        mRenderCtx.foundTransform = true;
        mRenderCtx.alignmentModel->setTransform(mRenderCtx.headTransform);
        mRenderCtx.expandedAlignmentModel->setTransform(mRenderCtx.headTransform * glm::scale(glm::mat4(), glm::vec3(1.1f, 1.1f, 1.1f)));
        mRenderCtx.model->setTransform(mRenderCtx.headTransform);
    }

    // Warp depth textures for occlusion
    mRealsenseDepth->warpToPair(mDepthFrame, mZedCalib, mRenderCtx.eyeMatrix[0], mRenderCtx.eyeMatrix[1]);

    // Render scene
    mOutputCtx->renderScene(mRenderCtx);
}

void RiftAR::keyEvent(int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_SPACE)
        {
            DebugOutput* debug = dynamic_cast<DebugOutput*>(mOutputCtx);
            if (debug)
                debug->toggleDebug();
        }

        if (key == GLFW_KEY_S)
        {
            mTracking->beginSearchingFor(mRenderCtx.alignmentModel);
            mRenderCtx.lookingForHead = true;
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

cv::Size RiftAR::getSize()
{
    return cv::Size(1600, 600);
}

// Distortion
void RiftAR::setupDepthWarpStream(cv::Size destinationSize)
{
    mRealsenseDepth = new RealsenseDepthAdjuster(mRealsense, destinationSize);
    mRenderCtx.depthTextures[0] = mRealsenseDepth->getDepthTexture(0);
    mRenderCtx.depthTextures[1] = mRealsenseDepth->getDepthTexture(1);

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
    mRenderCtx.eyeMatrix[0] = mRealsenseToZedLeft;
    mRenderCtx.eyeMatrix[1] = mZed->getExtrinsics(ZEDCamera::LEFT, ZEDCamera::RIGHT) * mRealsenseToZedLeft;
#else
    mRenderCtx.eyeMatrix[0] = mRenderCtx.eyeMatrix[1] = mRealsenseToZedLeft;
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
