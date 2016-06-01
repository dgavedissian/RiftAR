#pragma once

#include <chrono>
#include <thread>
#include <mutex>

#include <OVR_CAPI.h>

#include "lib/Rectangle2D.h"
#include "lib/Model.h"
#include "lib/Shader.h"

#include "camera/RealsenseCamera.h"
#include "camera/ZEDCamera.h"

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

    // Capture loop
    void captureLoop();
    bool getFrame();

    ZEDCamera* mZed;
    RealsenseCamera* mRealsense;

    // KFusion
    KFusionTracker* mTracking;

    // Warp parameters
    cv::Mat mDepthFrame;
    RealsenseDepthAdjuster* mRealsenseDepth;
    glm::mat3 mZedCalib;
    glm::mat4 mRealsenseToZedLeft;

    // Rendering
    RenderContext mRenderCtx;
    OutputContext* mOutputCtx;

    // Pose state
    int mFrameIndex;
    ovrPosef mEyePose[2];

    // Capture thread
    bool mHasFrame;
    bool mIsCapturing;
    std::thread mCaptureThread;
    std::mutex mFrameMutex;
    cv::Mat mDepth;

    // Configuration
    bool mAddArtificalLatency;

};
