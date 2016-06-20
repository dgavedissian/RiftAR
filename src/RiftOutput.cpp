#include "Common.h"
#include "RiftOutput.h"

RiftOutput::RiftOutput(cv::Size backbufferSize, uint cameraWidth, uint cameraHeight, float cameraFovH, bool invertColour) :
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

    // Get HMD to eye offsets
    ovrEyeRenderDesc eyeRenderDesc[2];
    eyeRenderDesc[0] = ovr_GetRenderDesc(mSession, ovrEye_Left, mHmdDesc.DefaultEyeFov[0]);
    eyeRenderDesc[1] = ovr_GetRenderDesc(mSession, ovrEye_Right, mHmdDesc.DefaultEyeFov[1]);
    mHmdToEyeOffset[0] = eyeRenderDesc[0].HmdToEyeOffset;
    mHmdToEyeOffset[1] = eyeRenderDesc[1].HmdToEyeOffset;

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
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, mBufferSize.w, mBufferSize.h, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr));

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

    // Calculate headset FOV
    float ovrFovH = atanf(mHmdDesc.DefaultEyeFov[0].LeftTan) + atanf(mHmdDesc.DefaultEyeFov[0].RightTan);
    float ovrFovV = atanf(mHmdDesc.DefaultEyeFov[0].UpTan) + atanf(mHmdDesc.DefaultEyeFov[0].DownTan);

    // Calculate the width and height of the camera stream being displayed on the headset in GL coordinates
    mFrameSize.width = (int)(cameraFovH / ovrFovH * (mBufferSize.w / 2));
    mFrameSize.height = (int)(mFrameSize.width * (cameraHeight / (float)cameraWidth));
    float widthGL = (float)mFrameSize.width / (mBufferSize.w / 2);
    float heightGL = (float)mFrameSize.height / mBufferSize.h;

    // Calculate offset
    float offsetLensCenterX = (atanf(mHmdDesc.DefaultEyeFov[0].LeftTan) / ovrFovH) * 2.f - 1.f;
    float offsetLensCenterY = (atanf(mHmdDesc.DefaultEyeFov[0].UpTan) / ovrFovV) * 2.f - 1.f;

    // Create rendering primitives
    mFullscreenQuad = make_unique<Rectangle2D>(glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 1.0f));
    mRiftMirrorShader = make_shared<Shader>("../media/quad.vs", "../media/quad.fs");
}

RiftOutput::~RiftOutput()
{
    ovr_Destroy(mSession);
    ovr_Shutdown();
}

void RiftOutput::newFrame(int& frameIndex, ovrPosef poses[2])
{
    // A frame has been completed
    frameIndex++;

    // Get eye poses, feeding in correct IPD offset
    double frameTiming = ovr_GetPredictedDisplayTime(mSession, frameIndex);
    ovrTrackingState state = ovr_GetTrackingState(mSession, frameTiming, ovrTrue);
    ovr_CalcEyePoses(state.HeadPose.ThePose, mHmdToEyeOffset, poses);
}

void RiftOutput::setFramePose(int frameIndex, ovrPosef poses[2])
{
    mFrameIndex = frameIndex;
    mEyePose[0] = poses[0];
    mEyePose[1] = poses[1];
}

void RiftOutput::renderScene(Renderer* ctx, int hit)
{
    // Get texture swap index where we must draw our frame
    GLuint curTexId;
    int curIndex;
    ovr_GetTextureSwapChainCurrentIndex(mSession, mTextureChain, &curIndex);
    ovr_GetTextureSwapChainBufferGL(mSession, mTextureChain, curIndex, &curTexId);

    // Bind the frame buffer
    glBindFramebuffer(GL_FRAMEBUFFER, mFramebufferId);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, curTexId, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, mDepthBufferId, 0);

    // Clear the frame buffer
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Render for each Oculus eye the equivalent ZED image
    for (int i = 0; i < 2; i++)
    {
        int hitFactor = i == 0 ? hit / 2 : -hit / 2;
        cv::Vec2i viewportPosition(i == ovrEye_Left ? hitFactor : mBufferSize.w / 2 + hitFactor, 0);
        cv::Size viewportSize(mBufferSize.w / 2, mBufferSize.h);
        cv::Vec2i viewportMid = viewportPosition + cv::Vec2i(viewportSize.width, viewportSize.height) / 2;
        ctx->setViewport(viewportMid - cv::Vec2i(mFrameSize.width, mFrameSize.height) / 2, mFrameSize);
        ctx->renderScene(i);
    }

    // Commit changes to the textures so they get picked up in the next frame
    ovr_CommitTextureSwapChain(mSession, mTextureChain);

    // Submit the frame
    ovrLayerEyeFov layer;
    layer.Header.Type = ovrLayerType_EyeFov;
    layer.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;
    for (int i = 0; i < 2; ++i)
    {
        layer.ColorTexture[i] = mTextureChain;
        layer.Viewport[i] = OVR::Recti(i == ovrEye_Left ? 0 : mBufferSize.w / 2, 0, mBufferSize.w / 2, mBufferSize.h);
        layer.Fov[i] = mHmdDesc.DefaultEyeFov[i];
        layer.RenderPose[i] = mEyePose[i];
    }
    ovrLayerHeader* layers = &layer.Header;
    ovrResult result = ovr_SubmitFrame(mSession, mFrameIndex, nullptr, &layers, 1);
    if (OVR_FAILURE(result))
        THROW_ERROR("Failed to submit frame!");

    // Draw the mirror texture
    glViewport(0, 0, ctx->getBackbufferSize().width, ctx->getBackbufferSize().height);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mMirrorTextureId);
    mRiftMirrorShader->bind();
    mFullscreenQuad->render();
}
