#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <opencv2/opencv.hpp>

class CameraSource
{
public:
    virtual ~CameraSource() {}

    enum Eye
    {
        LEFT = 0,
        RIGHT = 1
    };
    
    virtual void capture() = 0;
    virtual void updateTextures() = 0;

    // Helper functions for other algorithms
    virtual void copyFrameIntoCudaImage(Eye e, cudaGraphicsResource* resource) = 0;
    virtual void copyFrameIntoCVImage(Eye e, cv::Mat* mat) = 0;
    virtual const void* getRawData(Eye e) = 0;

    // Accessors
    int getWidth() const { return mWidth; }
    int getHeight() const { return mHeight; }
    GLuint getTexture(Eye e) const { return mTexture[e]; }

protected:
    int mWidth, mHeight;
    GLuint mTexture[2];

};