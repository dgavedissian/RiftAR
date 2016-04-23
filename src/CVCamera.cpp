#include "Common.h"
#include "CVCamera.h"

CVCamera::CVCamera(int device)
{
    mCap = new cv::VideoCapture(device);
    if (!mCap->isOpened())
        throw std::runtime_error("ERROR: Unable to open camera device");

    mWidth = (int)mCap->get(CV_CAP_PROP_FRAME_WIDTH);
    mHeight = (int)mCap->get(CV_CAP_PROP_FRAME_HEIGHT);

    // Set up image
    glGenTextures(1, &mTextureID);
    glBindTexture(GL_TEXTURE_2D, mTextureID);
    TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mWidth, mHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

CVCamera::~CVCamera()
{
    delete mCap;
}

void CVCamera::bindAndUpdate()
{
    cv::Mat frame;
    *mCap >> frame;

    // Grab data
    // TODO: Use a pixel buffer object instead of glTexSubImage2D
    glBindTexture(GL_TEXTURE_2D, mTextureID);
    TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_BGR, GL_UNSIGNED_BYTE, frame.ptr()));
}