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
    Model* model;
};

class OutputContext
{
public:
    virtual ~OutputContext() {}
    virtual void renderScene(RenderContext& ctx) = 0;
};
