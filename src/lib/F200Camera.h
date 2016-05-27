#pragma once

#include <thread>
#include <mutex>
#include <librealsense/rs.hpp>

#include "CameraSource.h"

class F200Camera : public CameraSource
{
public:
    enum
    {
        ENABLE_COLOUR = 1,
        ENABLE_DEPTH = 2,
        ENABLE_INFRARED = 4,
    };

    enum
    {
        COLOUR,
        DEPTH,
        INFRARED,
        STREAM_COUNT
    };

    F200Camera(uint width, uint height, uint frameRate, uint streams);
    ~F200Camera();

    void capture() override;
    void updateTextures() override;
    void copyFrameIntoCVImage(uint camera, cv::Mat* mat) override;
    const void* getRawData(uint camera) override;

    CameraIntrinsics getIntrinsics(uint camera) const override;
    glm::mat4 getExtrinsics(uint camera1, uint camera2) const override;
    GLuint getTexture(uint camera) const override;

    float getDepthScale() { return mDevice->get_depth_scale(); }

private:
    rs::stream mapCameraToStream(uint camera) const;
    CameraIntrinsics buildIntrinsics(rs::intrinsics& intr) const;

    void captureLoop();

    bool mIsCapturing;
    std::thread* mCaptureThread;
    std::mutex mFrameAccessMutex;

    uint mWidth, mHeight, mFrameRate;
    int mEnabledStreams;
    rs::context* mContext;
    rs::device* mDevice;

    void initialiseDevice();

    cv::Mat mStreamData[STREAM_COUNT];
    GLuint mStreamTextures[STREAM_COUNT];
};