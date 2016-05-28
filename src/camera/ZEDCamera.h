#pragma once

#include <zed/Camera.hpp>

#include "CameraSource.h"

class ZEDCamera : public CameraSource
{
public:
    ZEDCamera(sl::zed::ZEDResolution_mode resolution, int fps);
    ~ZEDCamera();

    enum
    {
        LEFT,
        RIGHT
    };

    void capture() override;
    void updateTextures() override;
    void copyFrameIntoCudaImage(uint camera, cudaGraphicsResource* resource);
    void copyFrameIntoCVImage(uint camera, cv::Mat* mat) override;
    const void* getRawData(uint camera) override;

    CameraIntrinsics getIntrinsics(uint camera) const override;
    glm::mat4 getExtrinsics(uint camera1, uint camera2) const override;
    GLuint getTexture(uint camera) const override;

private:
    sl::zed::Camera* mCamera;
    GLuint mTexture[2];
    cudaGraphicsResource* mCudaImage[2];

    CameraIntrinsics mIntrinsics;

    sl::zed::SIDE mapCameraToSide(uint camera) const;
};