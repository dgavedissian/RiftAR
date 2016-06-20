#pragma once

#include "lib/TextureCV.h"

class Entity;
class Shader;
class Rectangle2D;
class KFusionTracker;

enum RendererState
{
    RS_COLOUR,
    RS_DEBUG_DEPTH,
    RS_DEBUG_KFUSION
};

enum AlignmentState
{
    AS_NOT_FOUND,
    AS_SEARCHING,
    AS_FOUND
};

class Renderer
{
public:
    Renderer(
        const glm::mat4& projection,
        float znear, float zfar,
        cv::Size backbufferSize,
        float depthScale,
        KFusionTracker* tracker,
        bool invertColour);
    ~Renderer();

    void setTextures(GLuint colourTextures[2], GLuint depthTextures[2]);
    void setExtrinsics(glm::mat4 extrToEye[2]);

    void setViewport(cv::Point pos, cv::Size size);
    void renderScene(int eye);

    void setState(RendererState rs);

    // Scene
    void beginSearchingFor(unique_ptr<Entity> entity);
    void setObjectFound(glm::mat4 transform);

    // Matrices
    void setViewMatrix(const glm::mat4& view);

    // Accessors
    cv::Size getBackbufferSize() const;
    float getZNear() const;
    float getZFar() const;
    Entity* getTargetEntity();

private:
    cv::Size mBackbufferSize;

    // Configuration
    RendererState mState;

    // Alignment
    AlignmentState mAlignmentState;
    unique_ptr<Entity> mTargetEntity;
    unique_ptr<Entity> mExpandedTargetEntity;

    // Scene
    float mZNear, mZFar;
    unique_ptr<Entity> mOverlay;

    // Camera
    GLuint mColourTextures[2];
    GLuint mDepthTextures[2];
    glm::mat4 mExtrToEye[2];
    glm::mat4 mView, mProjection;

    // 2D Rectangle
    Rectangle2D* mQuad;
    Shader* mFullscreenShader;
    Shader* mFullscreenWithDepthShader;

    // Debugging
    KFusionTracker* mTracker;
    TextureCV mTrackerDebug;

};