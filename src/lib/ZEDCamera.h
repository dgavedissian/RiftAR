#pragma once

#include <zed/Camera.hpp>

#include "CameraSource.h"

class ZEDCamera : public CameraSource
{
public:
    ZEDCamera();
    ~ZEDCamera();

    void capture() override;
    void updateTextures() override;
    void copyFrameIntoCudaImage(Eye e, cudaGraphicsResource* resource) override;
    void copyFrameIntoCVImage(Eye e, cv::Mat* mat) override;
    const void* getRawData(Eye e) override;

private:
    sl::zed::Camera* mCamera;
    cudaGraphicsResource* mCudaImage[2];
};