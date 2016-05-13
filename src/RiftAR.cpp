#include "lib/Common.h"
#include "RiftAR.h"

#include "RiftOutput.h"
#include "DebugOutput.h"

//#define RIFT_DISPLAY
//#define ENABLE_ZED

DEFINE_MAIN(RiftAR);

RiftAR::RiftAR()
{
}

void RiftAR::init()
{
    // Set up the cameras
#ifdef ENABLE_ZED
    mZed = new ZEDCamera(sl::zed::HD720, 60);
#endif
    mRealsense = new F200Camera(640, 480, 60, F200Camera::ENABLE_COLOUR | F200Camera::ENABLE_DEPTH);

    // Get the width/height of the output colour stream that the user sees
    cv::Size destinationSize;
#ifdef ENABLE_ZED
    destinationSize.width = mZed->getWidth(ZEDCamera::LEFT);
    destinationSize.height = mZed->getHeight(ZEDCamera::LEFT);
#else
    destinationSize.width = mRealsense->getWidth(F200Camera::COLOUR);
    destinationSize.height = mRealsense->getHeight(F200Camera::COLOUR);
#endif

    // Set up depth warping
    setupDepthWarpStream(destinationSize);

    // Set up scene
    mRenderCtx.backbufferSize = getSize();
    mRenderCtx.depthScale = USHRT_MAX * mRealsense->getDepthScale();
    mRenderCtx.znear = 0.01f;
    mRenderCtx.zfar = 10.0f;
    mRenderCtx.projection = glm::perspective(glm::radians(75.0f), (float)destinationSize.width / (float)destinationSize.height, mRenderCtx.znear, mRenderCtx.zfar);
    mRenderCtx.model = new Model("../media/meshes/skull.stl");
    mRenderCtx.model->setPosition(glm::vec3(-0.4f, -0.4f, -1.2f));

    // Set up output
#ifdef ENABLE_ZED
    float fovH = mZed->getIntrinsics(ZEDCamera::LEFT).fovH;
    float fovV = mZed->getIntrinsics(ZEDCamera::LEFT).fovV;
    mRenderCtx.colourTextures[0] = mZed->getTexture(ZEDCamera::LEFT);
    mRenderCtx.colourTextures[1] = mZed->getTexture(ZEDCamera::RIGHT);
#else
    float fovH = mRealsense->getIntrinsics(F200Camera::COLOUR).fovH;
    float fovV = mRealsense->getIntrinsics(F200Camera::COLOUR).fovV;
    mRenderCtx.colourTextures[0] = mRealsense->getTexture(F200Camera::COLOUR);
    mRenderCtx.colourTextures[1] = mRealsense->getTexture(F200Camera::COLOUR);
#endif

#ifdef RIFT_DISPLAY
    mOutputCtx = new RiftOutput(getSize(), fovH, fovV);
#else
    mOutputCtx = new DebugOutput(mRenderCtx);
#endif

    // Enable culling and depth testing
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
}

RiftAR::~RiftAR()
{
    delete mRenderCtx.model;

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

    // Build depth texture
#ifdef ENABLE_ZED
    RealsenseDepth->warpToPair(mZedCalib,
        mRealsenseToZedLeft,
        mZed->getExtrinsics(ZEDCamera::LEFT, ZEDCamera::RIGHT) * mRealsenseToZedLeft);
#else
    RealsenseDepth->warpToPair(mZedCalib, mRealsenseToZedLeft, mRealsenseToZedLeft);
#endif

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
    }
}

cv::Size RiftAR::getSize()
{
    return cv::Size(1600, 600);
}

// Distortion
void RiftAR::setupDepthWarpStream(cv::Size destinationSize)
{
    RealsenseDepth = new RealsenseDepthAdjuster(mRealsense, destinationSize);
    mRenderCtx.depthTextures[0] = RealsenseDepth->getDepthTexture(0);
    mRenderCtx.depthTextures[1] = RealsenseDepth->getDepthTexture(1);

    // Read parameters
#ifdef ENABLE_ZED
    mZedCalib = convertCVToMat3<double>(mZed->getIntrinsics(ZEDCamera::LEFT).cameraMatrix);
#else
    mZedCalib = convertCVToMat3<double>(mRealsense->getIntrinsics(F200Camera::COLOUR).cameraMatrix);
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
    glm::mat4 depthToColour = mRealsense->getExtrinsics(F200Camera::DEPTH, F200Camera::COLOUR);

    // Combined extrinsics mapping realsense depth to ZED left
    mRealsenseToZedLeft = realsenseColourToZedLeft * depthToColour;
}
