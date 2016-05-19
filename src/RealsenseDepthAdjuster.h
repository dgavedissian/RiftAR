#pragma once

#include "lib/F200Camera.h"

template <class T>
struct DeviceImage
{
    uint2 size;
    T* data;

    __device__ T& operator[](const uint2& pos)
    {
        return data[pos.x + size.x * pos.y];
    }
};

class RealsenseDepthAdjuster
{
public:
    RealsenseDepthAdjuster(F200Camera* realsense, cv::Size destinationSize);
    ~RealsenseDepthAdjuster();

    void warpToPair(cv::Mat& frame, const glm::mat3& destCalib, const glm::mat4& leftExtrinsics, const glm::mat4& rightExtrinsics);
    GLuint getDepthTexture(uint eye);

private:
    void allocCuda(cv::Size srcSize);
    void freeCuda();

    F200Camera* mRealsense;

    glm::mat3 mRealsenseCalibInverse;
    std::vector<double> mRealsenseDistortCoeffs;
    cv::Size mColourSize;

    GLuint mDepthTextures[2];

    DeviceImage<uint16_t> mSrcImage;
    DeviceImage<uint32_t> mTempImage[2];
    DeviceImage<uint16_t> mDestImage[2];
    uint16_t* mDestImageHost[2]; // host mapping of mDestImage[0/1]
};
