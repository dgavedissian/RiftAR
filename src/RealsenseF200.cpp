#include "Common.h"
#include "RealsenseF200.h"

RealsenseF200::RealsenseF200(int width, int height, float frameRate) :
    mWidth(width),
    mHeight(height)
{
    mContext = new rs::context();
    if (mContext->get_device_count() == 0)
    {
        throw std::runtime_error("Unable to detect the Realsense F200");
    }

    mDevice = mContext->get_device(0);
    mDevice->enable_stream(rs::stream::depth, width, height, rs::format::z16, frameRate);
    mDevice->start();

    // Set up image
    glGenTextures(1, &mTextureID);
    glBindTexture(GL_TEXTURE_2D, mTextureID);
    TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_UNSIGNED_SHORT, nullptr));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

RealsenseF200::~RealsenseF200()
{
    delete mContext;
}

void RealsenseF200::bindAndUpdate()
{
    mDevice->wait_for_frames();

    // Grab data
    // TODO: Use a pixel buffer object instead of glTexSubImage2D
    glBindTexture(GL_TEXTURE_2D, mTextureID);
    TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RED, GL_UNSIGNED_SHORT, mDevice->get_frame_data(rs::stream::depth)));
}