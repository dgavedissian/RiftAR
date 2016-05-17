#include "Common.h"
#include "CameraSource.h"

// Reference: https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
#undef near
#undef far
glm::mat4 CameraIntrinsics::buildGLProjection(float near, float far)
{
    glm::mat3 calib = convertCVToMat3<double>(cameraMatrix);

    glm::mat4 perspective(0.0f);
    perspective[0][0] = calib[0][0];
    perspective[1][1] = calib[1][1];
    perspective[2][0] = -calib[2][0];
    perspective[2][1] = -calib[2][1];
    perspective[2][2] = near + far;
    perspective[2][3] = -1.0f;
    perspective[3][2] = near * far;

    glm::mat4 ndc = glm::ortho(0.0f, (float)width, 0.0f, (float)height, near, far);

    return ndc * perspective;
}

uint CameraSource::getWidth(uint camera) const
{
    return getIntrinsics(camera).width;
}

uint CameraSource::getHeight(uint camera) const
{
    return getIntrinsics(camera).height;
}
