#include "lib/Common.h"
#include "RiftAR.h"

#include "RiftOutput.h"
#include "DebugOutput.h"

#include "KFusionTracker.h"

//#define RIFT_DISPLAY
//#define ENABLE_ZED

DEFINE_MAIN(RiftAR);

RiftAR::RiftAR()
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
    mRenderCtx.alignmentModel = new Model("../media/meshes/bob.stl");
    mRenderCtx.model = new Model("../media/meshes/graymatter.stl");

    // Set up output
#ifdef ENABLE_ZED
    float fovH = mZed->getIntrinsics(ZEDCamera::LEFT).fovH;
    float fovV = mZed->getIntrinsics(ZEDCamera::LEFT).fovV;
    mRenderCtx.colourTextures[0] = mZed->getTexture(ZEDCamera::LEFT);
    mRenderCtx.colourTextures[1] = mZed->getTexture(ZEDCamera::RIGHT);
    mRenderCtx.projection = mZed->getIntrinsics(ZEDCamera::LEFT).buildGLProjection(mRenderCtx.znear, mRenderCtx.zfar);
#else
    float fovH = mRealsense->getIntrinsics(RealsenseCamera::COLOUR).fovH;
    float fovV = mRealsense->getIntrinsics(RealsenseCamera::COLOUR).fovV;
    mRenderCtx.colourTextures[0] = mRealsense->getTexture(RealsenseCamera::COLOUR);
    mRenderCtx.colourTextures[1] = mRealsense->getTexture(RealsenseCamera::COLOUR);
    mRenderCtx.projection = mRealsense->getIntrinsics(RealsenseCamera::COLOUR).buildGLProjection(mRenderCtx.znear, mRenderCtx.zfar);
#endif

#ifdef ENABLE_ZED
    bool invertColours = true;
#else
    bool invertColours = false;
#endif
#ifdef RIFT_DISPLAY
    mOutputCtx = new RiftOutput(getSize(), fovH, fovV, invertColours);
#else
    mOutputCtx = new DebugOutput(mRenderCtx, invertColours);
#endif

    // Enable culling and depth testing
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
}

RiftAR::~RiftAR()
{
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

    // Update the textures
#ifdef ENABLE_ZED
    mZed->capture();
    mZed->updateTextures();
#endif
    mRealsense->capture();
    mRealsense->updateTextures();

    // Read depth texture data
    static cv::Mat frame;
    mRealsense->copyFrameIntoCVImage(RealsenseCamera::DEPTH, &frame);

    // Update the cameras pose
    mTracking->update(frame);
    mRenderCtx.view = glm::inverse(mTracking->getCameraPose());

    // Search for the head
    if (mTracking->checkTargetPosition(mRenderCtx.headTransform))
    {
        mRenderCtx.lookingForHead = false;
        mRenderCtx.foundTransform = true;
        mRenderCtx.alignmentModel->setTransform(mRenderCtx.headTransform);
        mRenderCtx.model->setTransform(mRenderCtx.headTransform);
    }

    // Warp depth textures for occlusion
    mRealsenseDepth->warpToPair(frame, mZedCalib, mRenderCtx.eyeMatrix[0], mRenderCtx.eyeMatrix[1]);

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

        if (key == GLFW_KEY_L)
        {
            mTracking->beginSearchingFor(mRenderCtx.alignmentModel);
            mRenderCtx.lookingForHead = true;
        }

        if (key == GLFW_KEY_R)
        {
            mTracking->reset();
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
    glm::mat4 realsenseColourToZedLeft = buildExtrinsic(
        glm::inverse(convertCVToMat3<double>(rotationMatrix)),
        -convertCVToVec3<double>(translation));
#else
    glm::mat4 realsenseColourToZedLeft;
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
