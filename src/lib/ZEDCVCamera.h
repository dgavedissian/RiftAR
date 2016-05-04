#pragma once

#include "CameraSource.h"

class ZEDCVCamera : public CameraSource
{
public:
    ZEDCVCamera(int device);
    ~ZEDCVCamera();

    enum
    {
        LEFT,
        RIGHT
    };

    void capture() override;
    void updateTextures() override;
    void copyFrameIntoCudaImage(uint camera, cudaGraphicsResource* resource) override;
    void copyFrameIntoCVImage(uint camera, cv::Mat* mat) override;
    const void* getRawData(uint camera) override;

    CameraIntrinsics getIntrinsics(uint camera) override;
    CameraExtrinsics getExtrinsics(uint camera1, uint camera2) override;
    GLuint getTexture(uint camera) const override;

private:
    cv::VideoCapture* mCap;
    GLuint mTexture[2];
    cv::Mat mFrame[2];

    uint mWidth, mHeight;

    void checkCamera(uint camera) const;
};