#pragma once

#include "lib/TextureCV.h"

class Entity;
class Shader;
class Rectangle2D;
class KFusionTracker;

// TODO:
// - Write helper functions to make construction convenient
// - Make members private

enum RendererState
{
    RS_COLOUR,
    RS_DEBUG_DEPTH,
    RS_DEBUG_KFUSION
};

class Renderer
{
public:
    Renderer(bool invertColour, float znear, float zfar, float depthScale, KFusionTracker* tracker);
    ~Renderer();

    void setViewport(cv::Point pos, cv::Size size);
    void renderScene(int eye);

    void setState(RendererState rs);

    void toggleModel() { mShowModelAfterAlignment = !mShowModelAfterAlignment; }

//private:
    cv::Size backbufferSize;

    // Configuration
    RendererState mState;
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

    // Debugging
    KFusionTracker* mTracker;
    TextureCV mTrackerDebug;
};