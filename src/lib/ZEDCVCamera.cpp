#include "Common.h"
#include "ZEDCVCamera.h"

ZEDCVCamera::ZEDCVCamera(int device)
{
    mCap = new cv::VideoCapture(device);
    if (!mCap->isOpened())
        throw std::runtime_error("ERROR: Unable to open camera device");

    mWidth = (uint)mCap->get(CV_CAP_PROP_FRAME_WIDTH) / 2;
    mHeight = (uint)mCap->get(CV_CAP_PROP_FRAME_HEIGHT);

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

ZEDCVCamera::~ZEDCVCamera()
{
    delete mCap;
}

void ZEDCVCamera::capture()
{
    cv::Mat data;
    *mCap >> data;
    data.colRange(0, mWidth).copyTo(mFrame[LEFT]);
    data.colRange(mWidth, mWidth * 2).copyTo(mFrame[RIGHT]);
}

void ZEDCVCamera::updateTextures()
{
    // Copy each sub-image into two different GL images
    glBindTexture(GL_TEXTURE_2D, mTexture[0]);
    TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_BGR, GL_UNSIGNED_BYTE, getRawData(LEFT)));
    glBindTexture(GL_TEXTURE_2D, mTexture[1]);
    TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_BGR, GL_UNSIGNED_BYTE, getRawData(RIGHT)));
}

void ZEDCVCamera::copyFrameIntoCudaImage(uint camera, cudaGraphicsResource* resource)
{
    THROW_ERROR("Unimplemented");
}

void ZEDCVCamera::copyFrameIntoCVImage(uint camera, cv::Mat* mat)
{
    checkCamera(camera);
    mFrame[camera].copyTo(*mat);
}

const void* ZEDCVCamera::getRawData(uint camera)
{
    checkCamera(camera);
    return mFrame[camera].ptr();
}

CameraIntrinsics ZEDCVCamera::getIntrinsics(uint camera)
{
    return CameraIntrinsics();
}

CameraExtrinsics ZEDCVCamera::getExtrinsics(uint camera1, uint camera2)
{
    return CameraExtrinsics();
}

GLuint ZEDCVCamera::getTexture(uint camera) const
{
    checkCamera(camera);
    return mTexture[camera];
}

void ZEDCVCamera::checkCamera(uint camera) const
{
    if (camera > 1)
        THROW_ERROR("Camera must be ZEDCVCamera::LEFT or ZEDCVCamera::Right");
}
