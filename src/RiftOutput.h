#pragma once

#include "lib/Rectangle2D.h"
#include "lib/Shader.h"

#include "OutputContext.h"

#include <OVR_CAPI.h>
#include <OVR_CAPI_GL.h>
#include <Extras/OVR_Math.h>

class RiftOutput : public OutputContext
{
public:
    RiftOutput(cv::Size backbufferSize, float cameraFovH, float cameraFovV);
    ~RiftOutput();

    void renderScene(RenderContext& ctx) override;

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

    int mFrameIndex;

    Rectangle2D* mQuad;
    Rectangle2D* mFullscreenQuad;
    Shader* mFullscreenShader;
    Shader* mRiftMirrorShader;

};