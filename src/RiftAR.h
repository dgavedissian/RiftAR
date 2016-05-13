#pragma once

#include "lib/F200Camera.h"
#include "lib/ZEDCamera.h"

#include "lib/Rectangle2D.h"
#include "lib/Model.h"
#include "lib/Shader.h"

#include "OutputContext.h"
#include "RealsenseDepthAdjuster.h"

class RiftAR : public App
{
public:
    RiftAR();
    ~RiftAR();

    void init() override;
    void render() override;
    void keyEvent(int key, int scancode, int action, int mods) override;
    cv::Size getSize() override;

private:
    void setupDepthWarpStream(cv::Size destinationSize);

    ZEDCamera* mZed;
    F200Camera* mRealsense;

    // Warp parameters
    RealsenseDepthAdjuster* RealsenseDepth;
    glm::mat3 mZedCalib;
    glm::mat4 mRealsenseToZedLeft;

    // Rendering
    RenderContext mRenderCtx;
    OutputContext* mOutputCtx;

};
