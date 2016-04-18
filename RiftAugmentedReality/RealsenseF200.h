#pragma once

#include <librealsense/rs.hpp>

// RealSense F200 interface
class RealsenseF200
{
public:
    RealsenseF200(int width, int height, float frameRate);
    ~RealsenseF200();

    void bindAndUpdate();

    GLuint getTexture() const { return mTextureID; }

private:
    int mWidth;
    int mHeight;
    rs::context* mContext;
    rs::device* mDevice;
    GLuint mTextureID;
};