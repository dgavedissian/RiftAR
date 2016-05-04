#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

struct CameraIntrinsics
{
    int width, height;
    float fovH, fovV;

    // Format camera matrix and coeffs in the format that OpenCV expects
    cv::Mat cameraMatrix;
    std::vector<double> coeffs;
};

struct CameraExtrinsics
{
    glm::mat3 rotation;
    glm::vec3 translation;

    static CameraExtrinsics combine(CameraExtrinsics& a, CameraExtrinsics& b);
};

class CameraSource
{
public:
    virtual ~CameraSource() {}
    
    virtual void capture() = 0;
    virtual void updateTextures() = 0;

    // Helper functions for other algorithms
    virtual void copyFrameIntoCudaImage(uint camera, cudaGraphicsResource* resource) = 0;
    virtual void copyFrameIntoCVImage(uint camera, cv::Mat* mat) = 0;
    virtual const void* getRawData(uint camera) = 0;

    // Accessors
    virtual CameraIntrinsics getIntrinsics(uint camera) = 0;
    virtual CameraExtrinsics getExtrinsics(uint camera1, uint camera2) = 0;
    virtual GLuint getTexture(uint camera) const = 0;

};