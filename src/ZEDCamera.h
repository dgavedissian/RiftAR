#pragma once

#include <zed/Camera.hpp>

class ZEDCamera
{
public:
    ZEDCamera();
    ~ZEDCamera();

    void bindAndUpdate();

    GLuint getTexture() const { return mTextureID; }

private:
    sl::zed::Camera* mCamera;
    int mWidth;
    int mHeight;
    GLuint mTextureID;
};