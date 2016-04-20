#include "Common.h"
#include "ZEDCamera.h"

ZEDCamera::ZEDCamera()
{
    mCamera = new sl::zed::Camera(sl::zed::HD720);
    mWidth = mCamera->getImageSize().width;
    mHeight = mCamera->getImageSize().height;
    sl::zed::ERRCODE zederror = mCamera->init(sl::zed::MODE::PERFORMANCE, 0);
    if (zederror != sl::zed::SUCCESS)
    {
        throw std::runtime_error("ZED camera not detected");
    }

    // Set up image
    glGenTextures(1, &mTextureID);
    glBindTexture(GL_TEXTURE_2D, mTextureID);
    TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mWidth, mHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

ZEDCamera::~ZEDCamera()
{
    delete mCamera;
}

void ZEDCamera::bindAndUpdate()
{
    if (mCamera->grab(sl::zed::SENSING_MODE::RAW, false, false))
    {
        throw std::runtime_error("Error capturing frame from ZED camera");
    }

    // Grab data
    // TODO: Use a pixel buffer object instead of glTexSubImage2D
    glBindTexture(GL_TEXTURE_2D, mTextureID);
    TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_BGRA, GL_UNSIGNED_BYTE, mCamera->retrieveImage(sl::zed::SIDE::LEFT).data));
}