#pragma once

#include <opencv2/opencv.hpp>

#include "CameraSource.h"

class ZEDOpenCVCamera : public CameraSource
{
public:
    ZEDOpenCVCamera(int device);
    ~ZEDOpenCVCamera();

    void capture() override;
    void updateTextures() override;
    void copyFrameIntoCudaImage(Eye e, cudaGraphicsResource* resource) override;
    void copyFrameIntoCVImage(Eye e, cv::Mat* mat) override;
    const void* getRawData(Eye e) override;

private:
    cv::VideoCapture* mCap;
    cv::Mat mFrame[2];
};