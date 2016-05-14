#include "lib/Common.h"
#include "RiftOutput.h"

RiftOutput::RiftOutput(cv::Size backbufferSize, float cameraFovH, float cameraFovV, bool invertColour) :
    mFrameIndex(0)
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
    descMirrorTexture.Width = backbufferSize.width;
    descMirrorTexture.Height = backbufferSize.height;
    descMirrorTexture.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
    result = ovr_CreateMirrorTextureGL(mSession, &descMirrorTexture, &mMirrorTexture);
    if (!OVR_SUCCESS(result))
        THROW_ERROR("Failed to create the mirror texture");
    ovr_GetMirrorTextureBufferGL(mSession, mMirrorTexture, &mMirrorTextureId);

    // Calculate correct dimensions
    float ovrFovH = atanf(mHmdDesc.DefaultEyeFov[0].LeftTan) + atanf(mHmdDesc.DefaultEyeFov[0].RightTan);
    float ovrFovV = atanf(mHmdDesc.DefaultEyeFov[0].UpTan) + atanf(mHmdDesc.DefaultEyeFov[0].DownTan);
    float width = cameraFovH / ovrFovH;
    float height = cameraFovV / ovrFovV;

    // Create rendering primitives
    mQuad = new Rectangle2D(
        glm::vec2(0.5f - width * 0.5f, 0.5f - height * 0.5f),
        glm::vec2(0.5f + width * 0.5f, 0.5f + height * 0.5f));
    mFullscreenQuad = new Rectangle2D(glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 1.0f));
    mFullscreenShader = new Shader("../media/quad.vs", "../media/quad.fs");
    mFullscreenShader->setUniform("invertColour", invertColour);
    mRiftMirrorShader = new Shader("../media/quad.vs", "../media/quad.fs");
}

RiftOutput::~RiftOutput()
{
    delete mRiftMirrorShader;
    ovr_Destroy(mSession);
    ovr_Shutdown();
}

void RiftOutput::renderScene(RenderContext& ctx)
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
        glBindTexture(GL_TEXTURE_2D, ctx.colourTextures[i]);
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

    // Draw the mirror texture
    glViewport(0, 0, ctx.backbufferSize.width, ctx.backbufferSize.height);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, mMirrorTextureId);
    mRiftMirrorShader->bind();
    mFullscreenQuad->render();
}
