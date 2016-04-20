#pragma once

#include <librealsense/rs.hpp>

class F200Camera
{
public:
    F200Camera(int width, int height, float frameRate);
    ~F200Camera();

    void bindAndUpdate();

    GLuint getTexture() const { return mTextureID; }

private:
    int mWidth;
    int mHeight;
    rs::context* mContext;
    rs::device* mDevice;
    GLuint mTextureID;
};