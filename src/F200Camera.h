#pragma once

#include <librealsense/rs.hpp>

#include "CameraSource.h"

class F200CameraColour : public CameraSource
{
public:
    F200CameraColour(int width, int height, int frameRate);
    ~F200CameraColour();

    void capture() override;
    void updateTextures() override;
    void copyFrameIntoCudaImage(Eye e, cudaGraphicsResource* resource) override;
    void copyFrameIntoCVImage(Eye e, cv::Mat* mat) override;
    const void* getRawData(Eye e) override;

private:
    rs::context* mContext;
    rs::device* mDevice;
};

class F200CameraDepth : public CameraSource
{
public:
    F200CameraDepth(int width, int height, int frameRate);
    ~F200CameraDepth();

    void capture() override;
    void updateTextures() override;
    void copyFrameIntoCudaImage(Eye e, cudaGraphicsResource* resource) override;
    void copyFrameIntoCVImage(Eye e, cv::Mat* mat) override;
    const void* getRawData(Eye e) override;

private:
    rs::context* mContext;
    rs::device* mDevice;
};