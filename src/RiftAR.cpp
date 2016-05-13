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

    // Set up depth warping
    setupDepthWarpStream();

    // Set up scene
    mRenderCtx.backbufferSize = getSize();
    mRenderCtx.depthScale = USHRT_MAX * mRealsense->getDepthScale();
    mRenderCtx.znear = 0.01f;
    mRenderCtx.zfar = 10.0f;
    mRenderCtx.projection = glm::perspective(glm::radians(75.0f), (float)mColourSize.width / (float)mColourSize.height, mRenderCtx.znear, mRenderCtx.zfar);
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
    updateDepthTextures();

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
void RiftAR::setupDepthWarpStream()
{
    // Get the width/height of the output colour stream that the user sees
#ifdef ENABLE_ZED
    mColourSize.width = mZed->getWidth(ZEDCamera::LEFT);
    mColourSize.height = mZed->getHeight(ZEDCamera::LEFT);
#else
    mColourSize.width = mRealsense->getWidth(F200Camera::COLOUR);
    mColourSize.height = mRealsense->getHeight(F200Camera::COLOUR);
#endif

    // Create OpenGL images to view the depth stream
    glGenTextures(2, mRenderCtx.depthTextures);
    for (int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, mRenderCtx.depthTextures[i]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
            mColourSize.width, mColourSize.height,
            0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    // Read parameters
    CameraIntrinsics& depthIntr = mRealsense->getIntrinsics(F200Camera::DEPTH);
    mRealsenseCalibInverse = glm::inverse(convertCVToMat3<double>(depthIntr.cameraMatrix));
    mRealsenseDistortCoeffs = depthIntr.coeffs;
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

void RiftAR::updateDepthTextures()
{
    static cv::Mat frame, warpedFrame[2];

    // Copy depth frame from realsense and initialise warped frames for each eye
    mRealsense->copyFrameIntoCVImage(F200Camera::DEPTH, &frame);
    warpedFrame[0] = cv::Mat::zeros(cv::Size(mColourSize.width, mColourSize.height), CV_16UC1);
    warpedFrame[1] = cv::Mat::zeros(cv::Size(mColourSize.width, mColourSize.height), CV_16UC1);

    // Transform each pixel from the original frame using intrinsics and extrinsics
    glm::vec2 point;
    glm::mat4 realsenseToZedEye[2];
    realsenseToZedEye[0] = mRealsenseToZedLeft;
#ifdef ENABLE_ZED
    realsenseToZedEye[1] = mZed->getExtrinsics(ZEDCamera::LEFT, ZEDCamera::RIGHT) * mRealsenseToZedLeft;
#else
    realsenseToZedEye[1] = mRealsenseToZedLeft;
#endif
    for (int row = 0; row < frame.rows; row++)
    {
        uint16_t* rowData = frame.ptr<uint16_t>(row);
        for (int col = 0; col < frame.cols; col++)
        {
            // Read depth
            uint16_t depthPixel = rowData[col];
            if (depthPixel == 0)
                continue;
            float depth = depthPixel * mRealsense->getDepthScale();
            float newDepth;

            // Warp to each eye
            for (int i = 0; i < 2; i++)
            {
                // Top left of depth pixel
                point = glm::vec2((float)col - 0.5f, (float)row - 0.5f);
                newDepth = reprojectRealsenseToZed(point, depth, realsenseToZedEye[i]);
                cv::Point start((int)(point.x + 0.5f), (int)(point.y + 0.5f));

                // Bottom right of depth pixel
                point = glm::vec2((float)col + 0.5f, (float)row + 0.5f);
                newDepth = reprojectRealsenseToZed(point, depth, realsenseToZedEye[i]);
                cv::Point end((int)(point.x + 0.5f), (int)(point.y + 0.5f));

                // Swap start/end if appropriate
                if (start.x > end.x)
                    std::swap(start.x, end.x);
                if (start.y > end.y)
                    std::swap(start.y, end.y);

                // Reject pixels outside the target texture
                if (start.x < 0 || start.y < 0 || end.x >= warpedFrame[i].cols || end.y >= warpedFrame[i].rows)
                    continue;

                // Write the rectangle defined by the corners of the depth pixel to the output image
                for (int x = start.x; x <= end.x; x++)
                {
                    for (int y = start.y; y <= end.y; y++)
                        writeDepth(warpedFrame[i], x, y, newDepth);
                }
            }
        }
    }

    // Copy depth data
    for (int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, mRenderCtx.depthTextures[i]);
        TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mColourSize.width, mColourSize.height, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, warpedFrame[i].ptr()));
    }
}

float RiftAR::reprojectRealsenseToZed(glm::vec2& point, float depth, const glm::mat4& extrinsics)
{
    glm::vec3 homogenousPoint = glm::vec3(point, 1.0f);

    // De-project pixel to point in 3D space
    homogenousPoint.x = mRealsenseCalibInverse[0][0] * homogenousPoint.x + mRealsenseCalibInverse[2][0];
    homogenousPoint.y = mRealsenseCalibInverse[1][1] * homogenousPoint.y + mRealsenseCalibInverse[2][1];
    undistortRealsense(homogenousPoint, mRealsenseDistortCoeffs);
    homogenousPoint *= depth;

    // Map from Depth -> ZED
    homogenousPoint = glm::mat3(extrinsics) * homogenousPoint + glm::vec3(extrinsics[3]);

    // Project point to new pixel - conversion from vec4 to vec3 is equiv to multiplying by [I|0] matrix
    point.x = mZedCalib[0][0] * (homogenousPoint.x / homogenousPoint.z) + mZedCalib[2][0];
    point.y = mZedCalib[1][1] * (homogenousPoint.y / homogenousPoint.z) + mZedCalib[2][1];
    return homogenousPoint.z;
}

void RiftAR::writeDepth(cv::Mat& out, int x, int y, float depth)
{
    uint16_t oldDepth = out.at<uint16_t>(y, x);
    uint16_t newDepth = (uint16_t)(depth / mRealsense->getDepthScale());

    // Basic z-buffering here...
    if (newDepth < oldDepth || oldDepth == 0)
        out.at<uint16_t>(y, x) = newDepth;
}

void RiftAR::undistortRealsense(glm::vec3& point, const std::vector<double>& coeffs)
{
    float r2 = point.x * point.x + point.y * point.y;
    float f = 1.0f + coeffs[0] * r2 + coeffs[1] * r2 * r2 + coeffs[4] * r2 * r2 * r2;
    float ux = point.x * f + 2.0f * coeffs[2] * point.x * point.y + coeffs[3] * (r2 + 2.0f * point.x * point.x);
    float uy = point.y * f + 2.0f * coeffs[3] * point.x * point.y + coeffs[2] * (r2 + 2.0f * point.y * point.y);
    point.x = ux;
    point.y = uy;
}