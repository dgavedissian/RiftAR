#pragma once

#include "lib/F200Camera.h"
#include "lib/ZEDCamera.h"

#include "lib/Rectangle2D.h"
#include "lib/Model.h"
#include "lib/Shader.h"

#include "OutputContext.h"

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
    void setupDepthWarpStream();
    void updateDepthTextures();
    float reprojectRealsenseToZed(glm::vec2& point, float depth, const glm::mat4& extrinsics);
    void writeDepth(cv::Mat& out, int x, int y, float depth);
    void undistortRealsense(glm::vec3& point, const std::vector<double>& coeffs);

private:
    ZEDCamera* mZed;
    F200Camera* mRealsense;

    // Warp parameters
    glm::mat3 mRealsenseCalibInverse;
    std::vector<double> mRealsenseDistortCoeffs;
    glm::mat3 mZedCalib;
    glm::mat4 mRealsenseToZedLeft;
    cv::Size mColourSize;

    // Rendering
    RenderContext mRenderCtx;
    OutputContext* mOutputCtx;

};
