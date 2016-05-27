#pragma once

#include "lib/RealsenseCamera.h"
#include "lib/ZEDCamera.h"

#include "lib/Rectangle2D.h"
#include "lib/Model.h"
#include "lib/Shader.h"

#include "OutputContext.h"
#include "RealsenseDepthAdjuster.h"
#include "KFusionTracker.h"

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
    RealsenseCamera* mRealsense;

    // KFusion
    KFusionTracker* mTracking;

    // Warp parameters
    RealsenseDepthAdjuster* mRealsenseDepth;
    glm::mat3 mZedCalib;
    glm::mat4 mRealsenseToZedLeft;

    // Rendering
    RenderContext mRenderCtx;
    OutputContext* mOutputCtx;

};
