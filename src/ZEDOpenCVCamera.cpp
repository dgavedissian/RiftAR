#include "Common.h"
#include "ZEDOpenCVCamera.h"

ZEDOpenCVCamera::ZEDOpenCVCamera(int device)
{
    mCap = new cv::VideoCapture(device);
    if (!mCap->isOpened())
        throw std::runtime_error("ERROR: Unable to open camera device");

    mWidth = (int)mCap->get(CV_CAP_PROP_FRAME_WIDTH) / 2;
    mHeight = (int)mCap->get(CV_CAP_PROP_FRAME_HEIGHT);

    // Set up images
    glGenTextures(2, mTexture);
    glBindTexture(GL_TEXTURE_2D, mTexture[0]);
    TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mWidth, mHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, mTexture[1]);
    TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mWidth, mHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

ZEDOpenCVCamera::~ZEDOpenCVCamera()
{
    delete mCap;
}

void ZEDOpenCVCamera::capture()
{
    cv::Mat data;
    *mCap >> data;
    data.colRange(0, mWidth).copyTo(mFrame[LEFT]);
    data.colRange(mWidth, mWidth * 2).copyTo(mFrame[RIGHT]);
}

void ZEDOpenCVCamera::updateTextures()
{
    // Copy each sub-image into two different GL images
    glBindTexture(GL_TEXTURE_2D, mTexture[0]);
    TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_BGR, GL_UNSIGNED_BYTE, getRawData(LEFT)));
    glBindTexture(GL_TEXTURE_2D, mTexture[1]);
    TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_BGR, GL_UNSIGNED_BYTE, getRawData(RIGHT)));
}

void ZEDOpenCVCamera::copyFrameIntoCudaImage(Eye e, cudaGraphicsResource* resource)
{
    THROW_ERROR("Unimplemented");
}

void ZEDOpenCVCamera::copyFrameIntoCVImage(Eye e, cv::Mat* mat)
{
    mFrame[e].copyTo(*mat);
}

const void* ZEDOpenCVCamera::getRawData(Eye e)
{
    return mFrame[e].ptr();
}
