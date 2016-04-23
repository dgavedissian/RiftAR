#include "Common.h"
#include "F200Camera.h"

F200Camera::F200Camera(int width, int height, int frameRate, bool colourStream) :
    mColour(colourStream),
    mWidth(width),
    mHeight(height)
{
    mContext = new rs::context();
    if (mContext->get_device_count() == 0)
    {
        throw std::runtime_error("Unable to detect the RealsenseF200");
    }

    mDevice = mContext->get_device(0);
    if (mColour)
    {
        mDevice->enable_stream(rs::stream::color, width, height, rs::format::rgba8, frameRate);
    }
    else
    {
        mDevice->enable_stream(rs::stream::depth, width, height, rs::format::z16, frameRate);
    }
    mDevice->start();

    // Set up image
    glGenTextures(1, &mTexture[0]);
    glBindTexture(GL_TEXTURE_2D, mTexture[0]);
    if (mColour)
    {
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));
    }
    else
    {
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_UNSIGNED_SHORT, nullptr));
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    mTexture[1] = mTexture[0];
}

F200Camera::~F200Camera()
{
    delete mContext;
}

void F200Camera::retrieve()
{
    mDevice->wait_for_frames();

    // Grab data
    // TODO: Use a pixel buffer object instead of glTexSubImage2D?
    glBindTexture(GL_TEXTURE_2D, mTexture[0]);
    if (mColour)
    {
        TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, mDevice->get_frame_data(rs::stream::color)));
    }
    else
    {
        TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RED, GL_UNSIGNED_SHORT, mDevice->get_frame_data(rs::stream::depth)));
    }
}