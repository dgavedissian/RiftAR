#include "lib/Common.h"
#include "lib/F200Camera.h"
#include "lib/ZEDCamera.h"

#include "lib/Rectangle2D.h"
#include "lib/Shader.h"

#include <OVR_CAPI.h>
#include <OVR_CAPI_GL.h>
#include <Extras/OVR_Math.h>

class RiftAR : public App
{
public:
    RiftAR() :
        mFrameIndex(0)
    {
    }

    void init() override
    {
        ovrResult result = ovr_Initialize(nullptr);
        if (OVR_FAILURE(result))
            THROW_ERROR("Failed to initialise LibOVR");

        // Create a context for the rift device
        result = ovr_Create(&mSession, &mLuid);
        if (OVR_FAILURE(result))
            THROW_ERROR("Oculus Rift not detected");

        // Set up the cameras
        mZed = new ZEDCamera();
        mRealsense = new F200Camera(640, 480, 60, F200Camera::ENABLE_COLOUR | F200Camera::ENABLE_DEPTH);

        // Get the texture sizes of Oculus eyes
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

        // Calculate correct dimensions
        float ovrFovH = atanf(mHmdDesc.DefaultEyeFov[0].LeftTan) + atanf(mHmdDesc.DefaultEyeFov[0].RightTan);
        float ovrFovV = atanf(mHmdDesc.DefaultEyeFov[0].UpTan) + atanf(mHmdDesc.DefaultEyeFov[0].DownTan);
        float ratioWidth = mZed->getIntrinsics(ZEDCamera::LEFT).fovH / ovrFovH;
        float ratioHeight = mZed->getIntrinsics(ZEDCamera::LEFT).fovV / ovrFovV;
        float width = 1.0f * ratioWidth;
        float height = 1.0f * ratioHeight;

        // Create rendering primitives
        mQuad = new Rectangle2D(glm::vec2(0.5f - width * 0.5f, 0.5f - height * 0.5f), glm::vec2(0.5f + width * 0.5f, 0.5f + height * 0.5f));
        mQuadShader = new Shader("../media/quad.vs", "../media/quad_inv.fs");
        mMirrorShader = new Shader("../media/quad.vs", "../media/quad.fs");
    }

    ~RiftAR()
    {
        delete mQuad;
        delete mQuadShader;
        delete mMirrorShader;

        delete mZed;
        delete mRealsense;

        ovr_Destroy(mSession);
        ovr_Shutdown();
    }

    void render() override
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

        // Update the textures
        mZed->capture();
        mZed->updateTextures();

        // Bind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, mFramebufferId);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, curTexId, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, mDepthBufferId, 0);

        // Clear the frame buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0, 0, 0, 1);

        // Render for each Oculus eye the equivalent ZED image
        mQuadShader->bind();
        for (int eye = 0; eye < 2; eye++)
        {
            // Set the left or right vertical half of the buffer as the viewport
            glViewport(eye == ovrEye_Left ? 0 : mBufferSize.w / 2, 0, mBufferSize.w / 2, mBufferSize.h);

            // Bind the left or right ZED image
            glBindTexture(GL_TEXTURE_2D, mZed->getTexture(eye));
            mQuad->render();
        }

        // Commit changes to the textures so they get picked up frame
        ovr_CommitTextureSwapChain(mSession, mTextureChain);

        // Submit the frame
        ovrLayerEyeFov ld;
        ld.Header.Type = ovrLayerType_EyeFov;
        ld.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;
        for (int eye = 0; eye < 2; ++eye)
        {
            ld.ColorTexture[eye] = mTextureChain;
            ld.Viewport[eye] = OVR::Recti(eye == ovrEye_Left ? 0 : mBufferSize.w / 2, 0, mBufferSize.w / 2, mBufferSize.h);
            ld.Fov[eye] = mHmdDesc.DefaultEyeFov[eye];
            ld.RenderPose[eye] = eyeRenderPose[eye];
        }
        ovrLayerHeader* layers = &ld.Header;
        ovrResult result = ovr_SubmitFrame(mSession, mFrameIndex, nullptr, &layers, 1);
        if (OVR_FAILURE(result))
            THROW_ERROR("Failed to submit frame!");

        // Draw the mirror texture
        glViewport(0, 0, getSize().width, getSize().height);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, mMirrorTextureId);
        mMirrorShader->bind();
        mQuad->render();
        return;

        // A frame has been completed
        mFrameIndex++;
    }

    void keyEvent(int key, int scancode, int action, int mods) override
    {
    }

    cv::Size getSize() override
    {
        return cv::Size(1280, 720);
    }

private:
    ovrSession mSession;
    ovrGraphicsLuid mLuid;
    ovrHmdDesc mHmdDesc;
    ovrSizei mBufferSize;
    ovrTextureSwapChain mTextureChain;
    GLuint mFramebufferId;
    GLuint mDepthBufferId;
    ovrMirrorTexture mMirrorTexture;
    GLuint mMirrorTextureId;

    Rectangle2D* mQuad;
    Shader* mQuadShader;
    Shader* mMirrorShader;

    ZEDCamera* mZed;
    F200Camera* mRealsense;

    int mFrameIndex;

};

DEFINE_MAIN(RiftAR);
