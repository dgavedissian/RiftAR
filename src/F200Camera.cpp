#include "Common.h"
#include "F200Camera.h"

F200CameraColour::F200CameraColour(int width, int height, int frameRate)
{
    mWidth = width;
    mHeight = height;
    mContext = new rs::context();
    if (mContext->get_device_count() == 0)
        THROW_ERROR("Unable to detect the RealsenseF200");

    mDevice = mContext->get_device(0);
    mDevice->enable_stream(rs::stream::color, width, height, rs::format::rgba8, frameRate);
    mDevice->start();

    // Set up image
    glGenTextures(1, &mTexture[0]);
    glBindTexture(GL_TEXTURE_2D, mTexture[0]);
    TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    mTexture[1] = mTexture[0];
}

F200CameraColour::~F200CameraColour()
{
    delete mContext;
}

void F200CameraColour::capture()
{
    mDevice->wait_for_frames();
}

void F200CameraColour::updateTextures()
{
    // Grab data
    // TODO: Use a pixel buffer object instead of glTexSubImage2D?
    glBindTexture(GL_TEXTURE_2D, mTexture[0]);
    TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, getRawData(LEFT)));
}

void F200CameraColour::copyFrameIntoCudaImage(Eye e, cudaGraphicsResource* resource)
{
    THROW_ERROR("Unimplemented");
}

void F200CameraColour::copyFrameIntoCVImage(Eye e, cv::Mat* mat)
{
    // Wrap the data in a cv::Mat then copy it. This const_cast is sadly necessary as the
    // getRawData interface isn't supposed to allow writes to this location of memory.
    // As we immediately copy the data, it shouldn't matter much here.
    cv::Mat wrapped(mWidth, mHeight, CV_8UC4, const_cast<void*>(getRawData(e)));
    cvtColor(wrapped, *mat, cv::COLOR_RGBA2RGB);
}

const void* F200CameraColour::getRawData(Eye e)
{
    return mDevice->get_frame_data(rs::stream::color);
}

F200CameraDepth::F200CameraDepth(int width, int height, int frameRate)
{
    mWidth = width;
    mHeight = height;
    mContext = new rs::context();
    if (mContext->get_device_count() == 0)
        THROW_ERROR("Unable to detect the RealsenseF200");

    mDevice = mContext->get_device(0);
    mDevice->enable_stream(rs::stream::depth, width, height, rs::format::z16, frameRate);
    mDevice->start();

    // Set up image
    glGenTextures(1, &mTexture[0]);
    glBindTexture(GL_TEXTURE_2D, mTexture[0]);
    TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_UNSIGNED_SHORT, nullptr));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    mTexture[1] = mTexture[0];
}

F200CameraDepth::~F200CameraDepth()
{
    delete mContext;
}

void F200CameraDepth::capture()
{
    mDevice->wait_for_frames();
}

void F200CameraDepth::updateTextures()
{
    // Grab data
    // TODO: Use a pixel buffer object instead of glTexSubImage2D?
    glBindTexture(GL_TEXTURE_2D, mTexture[0]);
    TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RED, GL_UNSIGNED_SHORT, getRawData(LEFT)));
}

void F200CameraDepth::copyFrameIntoCudaImage(Eye e, cudaGraphicsResource* resource)
{
    THROW_ERROR("Unimplemented");
}

void F200CameraDepth::copyFrameIntoCVImage(Eye e, cv::Mat* mat)
{
    THROW_ERROR("Unimplemented");
}

const void* F200CameraDepth::getRawData(Eye e)
{
    return mDevice->get_frame_data(rs::stream::depth);
}