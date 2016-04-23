#pragma once

#include <opencv2/opencv.hpp>

#include "StereoCamera.h"

class ZEDOpenCVCamera : public StereoCamera
{
public:
    ZEDOpenCVCamera(int device);
    ~ZEDOpenCVCamera();

    void retrieve() override;

private:
    int mWidth;
    int mHeight;
    cv::VideoCapture* mCap;
};