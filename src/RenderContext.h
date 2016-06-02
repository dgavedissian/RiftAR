#pragma once

class Model;
class Shader;
class Rectangle2D;

// TODO: Make this into a class called "Renderer" and move camera frame rendering to here so Rift
// and Debug show the same thing
struct RenderContext
{
    cv::Size backbufferSize;

    // Configuration
    bool mShowColour;

    // Camera inputs
    GLuint colourTextures[2];
    GLuint depthTextures[2];

    // Scene
    float mZNear, mZFar;
    glm::mat4 view, projection;
    Model* alignmentModel;
    Model* expandedAlignmentModel; // TODO: probably worth allowing a model to share a mesh
    Model* model;

    // 2D Rectangle
    Rectangle2D* mQuad;
    Shader* mFullscreenShader;
    Shader* mFullscreenWithDepthShader;

    // Alignment
    bool lookingForHead;
    bool foundTransform;
    glm::mat4 headTransform;

    // Extrinsics
    glm::mat4 eyeMatrix[2];

    // Toggle showing depth
    void toggleDebug() { mShowColour = !mShowColour; }

    // Constructor
    RenderContext(bool invertColour, float znear, float zfar, float depthScale);
    ~RenderContext();

    // Render
    void setViewport(cv::Point pos, cv::Size size);
    void renderScene(int eye);
};