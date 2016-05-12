#include "lib/Common.h"
#include "RiftAR.h"

//#define RIFT_DISPLAY
//#define ENABLE_ZED

DEFINE_MAIN(RiftAR);

RiftAR::RiftAR() :
    mShowColour(true),
    mFrameIndex(0)
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
#ifdef ENABLE_ZED
    mColourSize.width = mZed->getWidth(ZEDCamera::LEFT);
    mColourSize.height = mZed->getHeight(ZEDCamera::LEFT);
#else
    mColourSize.width = mRealsense->getWidth(F200Camera::COLOUR);
    mColourSize.height = mRealsense->getHeight(F200Camera::COLOUR);
#endif

    // Create OpenGL images to view the depth stream
    glGenTextures(2, mDepthTexture);
    for (int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, mDepthTexture[i]);
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

#ifdef RIFT_DISPLAY
    // Set up Oculus
    setupOVR();

    // Calculate correct dimensions
    float ovrFovH = atanf(mHmdDesc.DefaultEyeFov[0].LeftTan) + atanf(mHmdDesc.DefaultEyeFov[0].RightTan);
    float ovrFovV = atanf(mHmdDesc.DefaultEyeFov[0].UpTan) + atanf(mHmdDesc.DefaultEyeFov[0].DownTan);
#ifdef ENABLE_ZED
    float width = mZed->getIntrinsics(ZEDCamera::LEFT).fovH / ovrFovH;
    float height = mZed->getIntrinsics(ZEDCamera::LEFT).fovV / ovrFovV;
#else
    float width = mRealsense->getIntrinsics(F200Camera::COLOUR).fovH / ovrFovH;
    float height = mRealsense->getIntrinsics(F200Camera::COLOUR).fovV / ovrFovV;
#endif

    // Create rendering primitives
    mQuad = new Rectangle2D(
        glm::vec2(0.5f - width * 0.5f, 0.5f - height * 0.5f),
        glm::vec2(0.5f + width * 0.5f, 0.5f + height * 0.5f));
    mFullscreenShader = new Shader("../media/quad.vs", "../media/quad_inv.fs");
    mRiftMirrorShader = new Shader("../media/quad.vs", "../media/quad.fs");
#else
    // Create rendering primitives
    mQuad = new Rectangle2D(glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 1.0f));
    mFullscreenShader = new Shader("../media/quad.vs", "../media/quad_inv.fs");
#endif
    // Create objects
    float znear = 0.01f;
    float zfar = 10.0f;
    mFullscreenWithDepthShader = new Shader("../media/quad.vs", "../media/quad_inv_depth.fs");
    mFullscreenWithDepthShader->bind();
    mFullscreenWithDepthShader->setUniform("rgbCameraImage", 0);
    mFullscreenWithDepthShader->setUniform("depthCameraImage", 1);
    mFullscreenWithDepthShader->setUniform("znear", znear);
    mFullscreenWithDepthShader->setUniform("zfar", zfar);
    mFullscreenWithDepthShader->setUniform("depthScale", USHRT_MAX * mRealsense->getDepthScale());

    mProjection = glm::perspective(glm::radians(75.0f), (float)mColourSize.width / (float)mColourSize.height, znear, zfar);
    mModel = new Model("../media/meshes/skull.stl");
    mModel->setPosition(glm::vec3(-0.4f, -0.4f, -1.2f));

    // Enable culling and depth testing
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
}

RiftAR::~RiftAR()
{
    delete mModel;
    delete mFullscreenWithDepthShader;
    delete mQuad;
    delete mFullscreenShader;

#ifdef ENABLE_ZED
    delete mZed;
#endif
    delete mRealsense;

#ifdef RIFT_DISPLAY
    delete mRiftMirrorShader;
    shutdownOVR();
#endif
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

#ifdef RIFT_DISPLAY
    // Render to the rift headset
    renderToRift();

    // Draw the mirror texture
    glViewport(0, 0, getSize().width, getSize().height);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, mMirrorTextureId);
    mRiftMirrorShader->bind();
    mQuad->render();
#else
    // Render each eye
    for (int i = 0; i < 2; i++)
    {
        glViewport(i == 0 ? 0 : getSize().width / 2, 0, getSize().width / 2, getSize().height);
        if (mShowColour)
        {
            glActiveTexture(GL_TEXTURE0);
#ifdef ENABLE_ZED
            glBindTexture(GL_TEXTURE_2D, mZed->getTexture(i));
#else
            glBindTexture(GL_TEXTURE_2D, mRealsense->getTexture(F200Camera::COLOUR));
#endif
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, mDepthTexture[i]);
            mFullscreenShader->bind();
            mQuad->render();

            mModel->render(mView, mProjection);
        }
        else
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, mDepthTexture[i]);
            mFullscreenShader->bind();
            mQuad->render();
        }
    }
#endif
}

void RiftAR::keyEvent(int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_SPACE)
            mShowColour = !mShowColour;
    }
}

cv::Size RiftAR::getSize()
{
    return cv::Size(1600, 600);
}

// Distortion
void RiftAR::updateDepthTextures()
{
    static cv::Mat frame, warpedFrame[2];

    // Copy depth frame from realsense and initialise warped frames for each eye
    mRealsense->copyFrameIntoCVImage(F200Camera::DEPTH, &frame);
    warpedFrame[0] = cv::Mat::zeros(cv::Size(mColourSize.width, mColourSize.height), CV_16UC1);
    warpedFrame[1] = cv::Mat::zeros(cv::Size(mColourSize.width, mColourSize.height), CV_16UC1);

    // Transform each pixel from the original frame using intrinsics and extrinsics
    glm::vec2 point;
#ifdef ENABLE_ZED
    glm::mat4 realsenseToCurrentZed = mZed->getExtrinsics(ZEDCamera::LEFT, i) * mRealsenseToZedLeft;
#else
    glm::mat4 realsenseToCurrentZed = mRealsenseToZedLeft;
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
                newDepth = reprojectRealsenseToZed(point, depth, realsenseToCurrentZed);
                cv::Point start((int)(point.x + 0.5f), (int)(point.y + 0.5f));

                // Bottom right of depth pixel
                point = glm::vec2((float)col + 0.5f, (float)row + 0.5f);
                newDepth = reprojectRealsenseToZed(point, depth, realsenseToCurrentZed);
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
        glBindTexture(GL_TEXTURE_2D, mDepthTexture[i]);
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

void RiftAR::setupOVR()
{
    ovrResult result = ovr_Initialize(nullptr);
    if (OVR_FAILURE(result))
        THROW_ERROR("Failed to initialise LibOVR");

    // Create a context for the rift device
    result = ovr_Create(&mSession, &mLuid);
    if (OVR_FAILURE(result))
        THROW_ERROR("Oculus Rift not detected");

    // Get the texture sizes of the rift eyes
    mHmdDesc = ovr_GetHmdDesc(mSession);
    ovrSizei textureSize0 = ovr_GetFovTextureSize(mSession, ovrEye_Left, mHmdDesc.DefaultEyeFov[0], 1.0f);
    ovrSizei textureSize1 = ovr_GetFovTextureSize(mSession, ovrEye_Right, mHmdDesc.DefaultEyeFov[1], 1.0f);

    // Compute the final size of the render buffer
    mBufferSize.w = textureSize0.w + textureSize1.w;
    mBufferSize.h = std::max(textureSize0.h, textureSize1.h);

    // Initialize OpenGL swap textures to render
    ovrTextureSwapChainDesc descTextureSwap = {};
    descTextureSwap.Type = ovrTexture_2D;
    descTextureSwap.ArraySize = 1;
    descTextureSwap.Width = mBufferSize.w;
    descTextureSwap.Height = mBufferSize.h;
    descTextureSwap.MipLevels = 1;
    descTextureSwap.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
    descTextureSwap.SampleCount = 1;
    descTextureSwap.StaticImage = ovrFalse;

    // Create the OpenGL texture swap chain, and enable linear filtering on each texture in the swap chain
    result = ovr_CreateTextureSwapChainGL(mSession, &descTextureSwap, &mTextureChain);
    int length = 0;
    ovr_GetTextureSwapChainLength(mSession, mTextureChain, &length);
    if (OVR_SUCCESS(result))
    {
        for (int i = 0; i < length; ++i)
        {
            GLuint chainTexId;
            ovr_GetTextureSwapChainBufferGL(mSession, mTextureChain, i, &chainTexId);
            glBindTexture(GL_TEXTURE_2D, chainTexId);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }
    }

    // Generate frame buffer to render
    glGenFramebuffers(1, &mFramebufferId);

    // Generate depth buffer of the frame buffer
    glGenTextures(1, &mDepthBufferId);
    glBindTexture(GL_TEXTURE_2D, mDepthBufferId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, mBufferSize.w, mBufferSize.h, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);

    // Create a mirror texture which is used to display the result in the GLFW window
    ovrMirrorTextureDesc descMirrorTexture;
    memset(&descMirrorTexture, 0, sizeof(descMirrorTexture));
    descMirrorTexture.Width = getSize().width;
    descMirrorTexture.Height = getSize().height;
    descMirrorTexture.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
    result = ovr_CreateMirrorTextureGL(mSession, &descMirrorTexture, &mMirrorTexture);
    if (!OVR_SUCCESS(result))
        THROW_ERROR("Failed to create the mirror texture");
    ovr_GetMirrorTextureBufferGL(mSession, mMirrorTexture, &mMirrorTextureId);
}

void RiftAR::shutdownOVR()
{
    ovr_Destroy(mSession);
    ovr_Shutdown();
}

void RiftAR::renderToRift()
{
    // Get texture swap index where we must draw our frame
    GLuint curTexId;
    int curIndex;
    ovr_GetTextureSwapChainCurrentIndex(mSession, mTextureChain, &curIndex);
    ovr_GetTextureSwapChainBufferGL(mSession, mTextureChain, curIndex, &curTexId);

    // Call ovr_GetRenderDesc each frame to get the ovrEyeRenderDesc, as the returned values (e.g. HmdToEyeOffset) may change at runtime.
    ovrEyeRenderDesc eyeRenderDesc[2];
    eyeRenderDesc[0] = ovr_GetRenderDesc(mSession, ovrEye_Left, mHmdDesc.DefaultEyeFov[0]);
    eyeRenderDesc[1] = ovr_GetRenderDesc(mSession, ovrEye_Right, mHmdDesc.DefaultEyeFov[1]);
    ovrVector3f hmdToEyeOffset[2];
    hmdToEyeOffset[0] = eyeRenderDesc[0].HmdToEyeOffset;
    hmdToEyeOffset[1] = eyeRenderDesc[1].HmdToEyeOffset;

    // Get eye poses, feeding in correct IPD offset
    ovrPosef eyeRenderPose[2];
    double sensorSampleTime;
    ovr_GetEyePoses(mSession, mFrameIndex, ovrTrue, hmdToEyeOffset, eyeRenderPose, &sensorSampleTime);

    // Bind the frame buffer
    glBindFramebuffer(GL_FRAMEBUFFER, mFramebufferId);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, curTexId, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, mDepthBufferId, 0);

    // Clear the frame buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0, 0, 0, 1);

    // Render for each Oculus eye the equivalent ZED image
    mFullscreenShader->bind();
    for (int i = 0; i < 2; i++)
    {
        // Set the left or right vertical half of the buffer as the viewport
        glViewport(i == ovrEye_Left ? 0 : mBufferSize.w / 2, 0, mBufferSize.w / 2, mBufferSize.h);

        // Bind the left or right ZED image
        glBindTexture(GL_TEXTURE_2D, mZed->getTexture(i));
        mQuad->render();
    }

    // Commit changes to the textures so they get picked up in the next frame
    ovr_CommitTextureSwapChain(mSession, mTextureChain);

    // Submit the frame
    ovrLayerEyeFov ld;
    ld.Header.Type = ovrLayerType_EyeFov;
    ld.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;
    for (int i = 0; i < 2; ++i)
    {
        ld.ColorTexture[i] = mTextureChain;
        ld.Viewport[i] = OVR::Recti(i == ovrEye_Left ? 0 : mBufferSize.w / 2, 0, mBufferSize.w / 2, mBufferSize.h);
        ld.Fov[i] = mHmdDesc.DefaultEyeFov[i];
        ld.RenderPose[i] = eyeRenderPose[i];
    }
    ovrLayerHeader* layers = &ld.Header;
    ovrResult result = ovr_SubmitFrame(mSession, mFrameIndex, nullptr, &layers, 1);
    if (OVR_FAILURE(result))
        THROW_ERROR("Failed to submit frame!");

    // A frame has been completed
    mFrameIndex++;
}