#pragma once

#include <chrono>
#include <thread>
#include <mutex>

#include <OVR_CAPI.h>

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
    void scrollEvent(double x, double y) override;
    cv::Size getSize() override;

private:
    // Capture loop
    void captureLoop();
    bool getFrame();

    // Cameras
    unique_ptr<ZEDCamera> mZed;
    unique_ptr<RealsenseCamera> mRealsense;

    // Extrinsics
    glm::mat4 mExtrRsToZed[2];

    // KFusion
    unique_ptr<KFusionTracker> mTracking;

    // Warp parameters
    cv::Mat mDepthFrame;
    glm::mat3 mDisplayIntr;
    unique_ptr<RealsenseDepthAdjuster> mRealsenseDepth;

    // Rendering
    unique_ptr<Renderer> mRenderer;
    unique_ptr<OutputContext> mOutputCtx;

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
