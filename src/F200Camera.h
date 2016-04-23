#pragma once

#include <librealsense/rs.hpp>

#include "StereoCamera.h"

class F200Camera : public StereoCamera
{
public:
    F200Camera(int width, int height, int frameRate, bool colourStream);
    ~F200Camera();

    void retrieve() override;

private:
    bool mColour;
    int mWidth;
    int mHeight;
    rs::context* mContext;
    rs::device* mDevice;
};