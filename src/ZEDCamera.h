#pragma once

#include <zed/Camera.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class ZEDCamera
{
public:
    ZEDCamera();
    ~ZEDCamera();

    void bindAndUpdate();

    GLuint getTexture() const { return mTextureID; }

private:
    int mWidth;
    int mHeight;
    sl::zed::Camera* mCamera;
    GLuint mTextureID;
    cudaGraphicsResource* mCudaImage;
};