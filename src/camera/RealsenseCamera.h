#pragma once

#include <librealsense/rs.hpp>

#include "CameraSource.h"

class RealsenseCamera : public CameraSource
{
public:
    enum
    {
        COLOUR,
        DEPTH,
        INFRARED,
        STREAM_COUNT
    };

    enum
    {
        ENABLE_COLOUR = 1 << COLOUR,
        ENABLE_DEPTH = 1 << DEPTH,
        ENABLE_INFRARED = 1 << INFRARED,
    };

    RealsenseCamera(uint width, uint height, uint frameRate, uint streams);
    ~RealsenseCamera();

    void capture() override;
    void copyData() override;
    void updateTextures() override;

    void copyFrameIntoCVImage(uint camera, cv::Mat* mat) override;
    const void* getRawData(uint camera) override;

    CameraIntrinsics getIntrinsics(uint camera) const override;
    glm::mat4 getExtrinsics(uint camera1, uint camera2) const override;
    GLuint getTexture(uint camera) const override;

    float getDepthScale() { return mDevice->get_depth_scale(); }
    bool isStreamEnabled(uint camera) const;

private:
    rs::stream mapCameraToStream(uint camera) const;
    CameraIntrinsics buildIntrinsics(uint camera) const;
    glm::mat4 buildExtrinsics(uint camera1, uint camera2) const;

    void initialiseDevice();

    uint mWidth, mHeight, mFrameRate;
    int mEnabledStreams;
    rs::context* mContext;
    rs::device* mDevice;

    cv::Mat mStreamData[STREAM_COUNT];
    GLuint mStreamTextures[STREAM_COUNT];
    CameraIntrinsics mIntrinsics[STREAM_COUNT];
    glm::mat4 mExtrinsics[STREAM_COUNT][STREAM_COUNT];
};