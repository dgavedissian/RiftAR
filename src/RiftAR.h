#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>

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
    void captureLoop();

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

    // Capture thread
    bool mIsCapturing;
    std::thread* mCaptureThread;
    std::mutex mCaptureLock;
    cv::Mat mDepth;

    // Condition variable to wait until capture thread has at least one frame
    std::condition_variable mCondVar;
    std::mutex mCondVarMutex;
    bool mInitialisedStream;

};
