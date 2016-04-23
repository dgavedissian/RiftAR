#pragma once

#include <opencv2/opencv.hpp>

class CVCamera
{
public:
    CVCamera(int device);
    ~CVCamera();

    void bindAndUpdate();

    GLuint getTexture() const { return mTextureID; }

private:
    int mWidth;
    int mHeight;
    cv::VideoCapture* mCap;
    GLuint mTextureID;
};