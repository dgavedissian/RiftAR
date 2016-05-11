#include "lib/Common.h"
#include "lib/F200Camera.h"
#include "lib/ZEDCamera.h"

#include "lib/Rectangle2D.h"
#include "lib/STLModel.h"
#include "lib/Shader.h"

#include <OVR_CAPI.h>
#include <OVR_CAPI_GL.h>
#include <Extras/OVR_Math.h>

//#define RIFT_DISPLAY

class RiftAR : public App
{
public:
    RiftAR() :
        mShowColour(false),
        mFrameIndex(0)
    {
    }

    void init() override
    {
        // Set up the cameras
        mZed = new ZEDCamera(sl::zed::HD720, 0);
        mRealsense = new F200Camera(640, 480, 60, F200Camera::ENABLE_DEPTH);

        // Create OpenGL images to view the depth stream
        glGenTextures(2, mDepthTexture);
        for (int i = 0; i < 2; i++)
        {
            glBindTexture(GL_TEXTURE_2D, mDepthTexture[i]);
            TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                mZed->getWidth(ZEDCamera::LEFT), mZed->getHeight(ZEDCamera::LEFT),
                0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, nullptr));
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }

        // Read parameters
        CameraIntrinsics& depthIntr = mRealsense->getIntrinsics(F200Camera::DEPTH);
        mRealsenseCalibInverse = glm::inverse(convertCVToMat3<double>(depthIntr.cameraMatrix));
        mRealsenseDistortCoeffs = depthIntr.coeffs;
        mZedCalib = convertCVToMat3<double>(mZed->getIntrinsics(ZEDCamera::LEFT).cameraMatrix);

        // Read extrinsics parameters that map the ZED to the realsense colour camera, and invert
        // to map in the opposite direction
        cv::FileStorage fs("../stereo-params.xml", cv::FileStorage::READ);
        cv::Mat rotationMatrix, translation;
        fs["R"] >> rotationMatrix;
        fs["T"] >> translation;
        glm::mat4 realsenseColourToZedLeft = buildExtrinsic(
            glm::inverse(convertCVToMat3<double>(rotationMatrix)),
            -convertCVToVec3<double>(translation));

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
        float width = mZed->getIntrinsics(ZEDCamera::LEFT).fovH / ovrFovH;
        float height = mZed->getIntrinsics(ZEDCamera::LEFT).fovV / ovrFovV;

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
        mQuad = new Rectangle2D(glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 1.0f));
        mFullscreenWithDepthShader = new Shader("../media/quad.vs", "../media/quad_inv_depth.fs");
        mFullscreenWithDepthShader->bind();
        mFullscreenWithDepthShader->setUniform("rgbCameraImage", 0);
        mFullscreenWithDepthShader->setUniform("depthCameraImage", 1);
        mFullscreenWithDepthShader->setUniform("znear", znear);
        mFullscreenWithDepthShader->setUniform("zfar", zfar);
        mFullscreenWithDepthShader->setUniform("depthScale", USHRT_MAX * mRealsense->getDepthScale());

        glm::mat4 model = glm::scale(glm::translate(glm::mat4(), glm::vec3(-0.4f, -0.4f, -1.2f)), glm::vec3(3.0f));
        glm::mat4 view;
        glm::mat4 projection = glm::perspective(glm::radians(75.0f), 640.0f / 480.0f, znear, zfar);
        mModel = new STLModel("../media/meshes/skull.stl");
        mModelShader = new Shader("../media/model.vs", "../media/model.fs");
        mModelShader->bind();
        mModelShader->setUniform("modelViewProjectionMatrix", projection * model);
        mModelShader->setUniform("modelMatrix", model);

        // Enable culling and depth testing
        glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
    }

    ~RiftAR()
    {
        delete mQuad;
        delete mFullscreenShader;

        delete mZed;
        delete mRealsense;

#ifdef RIFT_DISPLAY
        delete mRiftMirrorShader;
        shutdownOVR();
#endif
    }

    void render() override
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Update the textures
        mZed->capture();
        mZed->updateTextures();
        mRealsense->capture();

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
                glBindTexture(GL_TEXTURE_2D, mZed->getTexture(i));
                mFullscreenShader->bind();
                mQuad->render();
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

    void keyEvent(int key, int scancode, int action, int mods) override
    {
    }

    cv::Size getSize() override
    {
        return cv::Size(1600, 600);
    }

    // Distortion
    void updateDepthTextures()
    {
        cv::Mat frame, warpedFrame;
        for (int i = 0; i < 2; i++)
        {
            mRealsense->copyFrameIntoCVImage(F200Camera::DEPTH, &frame);

            // Create the output depth frame and initialise to maximum depth
            warpedFrame = cv::Mat::zeros(cv::Size(mZed->getWidth(ZEDCamera::LEFT), mZed->getHeight(ZEDCamera::RIGHT)), CV_16UC1);
            for (int c = 0; c < warpedFrame.cols; c++)
            {
                for (int r = 0; r < warpedFrame.rows; r++)
                    warpedFrame.at<uint16_t>(r, c) = 0xffff;
            }

            // Transform each pixel from the original frame using the camera matrices above
            glm::vec2 point;
            glm::mat4 realsenseToCurrentZed = mZed->getExtrinsics(ZEDCamera::LEFT, i) * mRealsenseToZedLeft;
            for (int row = 0; row < frame.rows; row++)
            {
                for (int col = 0; col < frame.cols; col++)
                {
                    // Read depth
                    uint16_t depthPixel = frame.at<uint16_t>(row, col);
                    if (depthPixel == 0)
                        continue;
                    float depth = (float)depthPixel * mRealsense->getDepthScale();
                    float newDepth;

                    // Top left of depth pixel
                    point = glm::vec2((float)col - 0.5f, (float)row - 0.5f);
                    newDepth = reprojectRealsenseToZed(point, depth, realsenseToCurrentZed);
                    cv::Point start((int)std::round(point.x), (int)std::round(point.y));

                    // Bottom right of depth pixel
                    point = glm::vec2((float)col + 0.5f, (float)row + 0.5f);
                    newDepth = reprojectRealsenseToZed(point, depth, realsenseToCurrentZed);
                    cv::Point end((int)std::round(point.x), (int)std::round(point.y));

                    // Swap start/end if appropriate
                    if (start.x > end.x)
                        std::swap(start.x, end.x);
                    if (start.y > end.y)
                        std::swap(start.y, end.y);

                    // Reject pixels outside the target texture
                    if (start.x < 0 || start.y < 0 || end.x >= warpedFrame.cols || end.y >= warpedFrame.rows)
                        continue;

                    // Write the rectangle defined by the corners of the depth pixel to the output image
                    for (int x = start.x; x <= end.x; x++)
                    {
                        for (int y = start.y; y <= end.y; y++)
                            writeDepth(warpedFrame, x, y, newDepth);
                    }
                }
            }

            // Copy depth data
            glBindTexture(GL_TEXTURE_2D, mDepthTexture[i]);
            TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                mZed->getWidth(ZEDCamera::LEFT), mZed->getHeight(ZEDCamera::LEFT),
                GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, warpedFrame.ptr()));
        }
    }

    float reprojectRealsenseToZed(glm::vec2& point, float depth, const glm::mat4& extrinsics)
    {
        glm::vec3 point2d;
        glm::vec4 point3d;

        point2d = glm::vec3(point, 1.0f);

        // De-project pixel to point and convert to 4D homogeneous coordinates
        point2d = mRealsenseCalibInverse * point2d;
        undistortRealsense(point2d, mRealsenseDistortCoeffs);
        point3d = glm::vec4(depth * point2d, 1.0f);

        // Map from Depth -> ZED
        point3d = extrinsics * point3d;

        // Project point - conversion from vec3 to vec3 is equiv to multiplying by [I|0] matrix
        point2d = mZedCalib * glm::vec3(point3d.x, point3d.y, point3d.z);

        // Record depth and convert to cartesian
        point.x = point2d.x / point2d.z;
        point.y = point2d.y / point2d.z;
        return point2d.z;
    }

    void writeDepth(cv::Mat& out, int x, int y, float depth)
    {
        uint16_t oldDepth = out.at<uint16_t>(y, x);
        uint16_t newDepth = (uint16_t)(depth / mRealsense->getDepthScale());

        // Basic z-buffering here...
        if (newDepth < oldDepth)
        {
            out.at<uint16_t>(y, x) = newDepth;
        }
    }

    void undistortRealsense(glm::vec3& point, const std::vector<double>& coeffs)
    {
        float r2 = point.x * point.x + point.y * point.y;
        float f = 1.0f + coeffs[0] * r2 + coeffs[1] * r2 * r2 + coeffs[4] * r2 * r2 * r2;
        float ux = point.x * f + 2.0f * coeffs[2] * point.x * point.y + coeffs[3] * (r2 + 2.0f * point.x * point.x);
        float uy = point.y * f + 2.0f * coeffs[3] * point.x * point.y + coeffs[2] * (r2 + 2.0f * point.y * point.y);
        point.x = ux;
        point.y = uy;
    }

    // Rift Interface
    void setupOVR()
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

    void shutdownOVR()
    {
        ovr_Destroy(mSession);
        ovr_Shutdown();
    }

    void renderToRift()
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

private:
    bool mShowColour;

    Rectangle2D* mQuad;
    Shader* mFullscreenShader;
    Shader* mFullscreenWithDepthShader;
    Shader* mRiftMirrorShader;

    STLModel* mModel;
    Shader* mModelShader;

    ZEDCamera* mZed;
    F200Camera* mRealsense;

    GLuint mDepthTexture[2];

    // Warp parameters
    glm::mat3 mRealsenseCalibInverse;
    std::vector<double> mRealsenseDistortCoeffs;
    glm::mat3 mZedCalib;
    glm::mat4 mRealsenseToZedLeft;

    // OVR stuff
    ovrSession mSession;
    ovrGraphicsLuid mLuid;
    ovrHmdDesc mHmdDesc;
    ovrSizei mBufferSize;
    ovrTextureSwapChain mTextureChain;
    GLuint mFramebufferId;
    GLuint mDepthBufferId;
    ovrMirrorTexture mMirrorTexture;
    GLuint mMirrorTextureId;

    int mFrameIndex;

};

DEFINE_MAIN(RiftAR);
