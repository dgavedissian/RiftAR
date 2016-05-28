#pragma once

struct CameraIntrinsics
{
    int width, height;
    float fovH, fovV;

    // Format camera matrix and coeffs in the format that OpenCV expects
    cv::Mat cameraMatrix;
    std::vector<double> coeffs;

    // Build an OpenGL projection matrix that uses these intrinsic parameters
    glm::mat4 buildGLProjection(float near, float far);
};

inline glm::mat4 buildExtrinsic(glm::mat3& rotation, glm::vec3& translation)
{
    return glm::translate(glm::mat4(rotation), translation);
}

class CameraSource
{
public:
    virtual ~CameraSource() {}
    
    virtual void capture() = 0;         // Capture frame data from the camera API
    virtual void copyData() = 0;        // Copy data from API to intermediate store
    virtual void updateTextures() = 0;  // Update OpenGL textures from intermediate store

    // Helper functions for other algorithms
    virtual void copyFrameIntoCVImage(uint camera, cv::Mat* mat) = 0;
    virtual const void* getRawData(uint camera) = 0;

    // Accessors
    virtual CameraIntrinsics getIntrinsics(uint camera) const = 0;
    virtual glm::mat4 getExtrinsics(uint camera1, uint camera2) const = 0;
    virtual GLuint getTexture(uint camera) const = 0;

    uint getWidth(uint camera) const;
    uint getHeight(uint camera) const;

};