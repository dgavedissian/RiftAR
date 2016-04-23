#pragma once

#include <zed/Camera.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "StereoCamera.h"

class ZEDCamera : public StereoCamera
{
public:
    ZEDCamera();
    ~ZEDCamera();

    void retrieve() override;

private:
    int mWidth;
    int mHeight;
    sl::zed::Camera* mCamera;
    cudaGraphicsResource* mCudaImage[2];
};