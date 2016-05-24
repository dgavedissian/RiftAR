#pragma once

class Model;

struct RenderContext
{
    cv::Size backbufferSize;

    // Camera inputs
    GLuint colourTextures[2];
    GLuint depthTextures[2];
    float depthScale;

    // Scene
    float znear, zfar;
    glm::mat4 view, projection;
    Model* alignmentModel;
    Model* model;

    // Alignment
    bool lookingForHead;
    bool foundTransform;
    glm::mat4 headTransform;

    // Extrinsics
    glm::mat4 eyeMatrix[2];

    // Render
    void renderScene(int eye);
};