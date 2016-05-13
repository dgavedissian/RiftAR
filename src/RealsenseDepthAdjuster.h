#pragma once

#include "lib/F200Camera.h"

class RealsenseDepthAdjuster
{
public:
    RealsenseDepthAdjuster(F200Camera* realsense, cv::Size destinationSize);
    ~RealsenseDepthAdjuster();

    void warpToPair(const glm::mat3& destCalib, const glm::mat4& leftExtrinsics, const glm::mat4& rightExtrinsics);
    GLuint getDepthTexture(uint eye);

private:
    float reprojectRealsenseToZed(glm::vec2& point, float depth, const glm::mat3& destCalib, const glm::mat4& extrinsics);
    void writeDepth(cv::Mat& out, int x, int y, float depth);
    void undistortRealsense(glm::vec3& point, const std::vector<double>& coeffs);

    F200Camera* mRealsense;

    glm::mat3 mRealsenseCalibInverse;
    std::vector<double> mRealsenseDistortCoeffs;
    cv::Size mColourSize;

    GLuint mDepthTextures[2];
};