#pragma once

#include "lib/F200Camera.h"
#include "lib/ZEDCamera.h"

#include "lib/Rectangle2D.h"
#include "lib/Model.h"
#include "lib/Shader.h"

#include <OVR_CAPI.h>
#include <OVR_CAPI_GL.h>
#include <Extras/OVR_Math.h>

//#define RIFT_DISPLAY
//#define ENABLE_ZED

class RiftAR : public App
{
public:
    RiftAR();
    ~RiftAR();

    void init() override;
    void render() override;
    void keyEvent(int key, int scancode, int action, int mods) override;
    cv::Size getSize() override;

    // Distortion
    void updateDepthTextures();
    float reprojectRealsenseToZed(glm::vec2& point, float depth, const glm::mat4& extrinsics);
    void writeDepth(cv::Mat& out, int x, int y, float depth);
    void undistortRealsense(glm::vec3& point, const std::vector<double>& coeffs);

    // Rift Interface
    void setupOVR();
    void shutdownOVR();
    void renderToRift();

private:
    bool mShowColour;

    Rectangle2D* mQuad;
    Shader* mFullscreenShader;
    Shader* mFullscreenWithDepthShader;
    Shader* mRiftMirrorShader;

    Model* mModel;
    glm::mat4 mView, mProjection;

    ZEDCamera* mZed;
    F200Camera* mRealsense;

    GLuint mDepthTexture[2];

    // Warp parameters
    glm::mat3 mRealsenseCalibInverse;
    std::vector<double> mRealsenseDistortCoeffs;
    glm::mat3 mZedCalib;
    glm::mat4 mRealsenseToZedLeft;
    cv::Size mColourSize;

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
