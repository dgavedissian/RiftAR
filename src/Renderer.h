#pragma once

class Entity;
class Shader;
class Rectangle2D;

// TODO:
// - Write helper functions to make construction convenient
// - Make members private

class Renderer
{
public:
    Renderer(bool invertColour, float znear, float zfar, float depthScale);
    ~Renderer();

    void setViewport(cv::Point pos, cv::Size size);
    void renderScene(int eye);

    // Toggles
    void toggleDebug() { mShowColour = !mShowColour; }
    void toggleModel() { mShowModelAfterAlignment = !mShowModelAfterAlignment; }

//private:
    cv::Size backbufferSize;

    // Configuration
    bool mShowColour;
    bool mShowModelAfterAlignment;

    // Camera inputs
    GLuint colourTextures[2];
    GLuint depthTextures[2];

    // Scene
    float mZNear, mZFar;
    glm::mat4 view, projection;
    Entity* alignmentEntity;
    Entity* expandedAlignmentEntity;
    Entity* overlay;

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
};