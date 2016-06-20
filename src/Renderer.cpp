#include "Common.h"
#include "Renderer.h"
#include "KFusionTracker.h"

#include "lib/Entity.h"
#include "lib/Model.h"
#include "lib/Shader.h"
#include "lib/Rectangle2D.h"
#include "lib/TextureCV.h"

Renderer::Renderer(const glm::mat4& projection, float znear, float zfar, cv::Size backbufferSize, float depthScale, KFusionTracker* tracker, bool invertColour) :
    mState(RS_COLOUR),
    mAlignmentState(AS_NOT_FOUND),
    mZNear(znear),
    mZFar(zfar),
    mBackbufferSize(backbufferSize),
    mTracker(tracker),
    mProjection(projection)
{
    // Create rendering primitives
    mQuad = make_unique<Rectangle2D>(glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 1.0f));
    mFullscreenShader = make_shared<Shader>("../media/quad.vs", "../media/quad.fs");
    mFullscreenShader->setUniform("invertColour", invertColour);

    // Create objects
    mFullscreenWithDepthShader = make_shared<Shader>("../media/quad.vs", "../media/quad_depth.fs");
    mFullscreenWithDepthShader->setUniform("invertColour", invertColour);
    mFullscreenWithDepthShader->setUniform("rgbCameraImage", 0);
    mFullscreenWithDepthShader->setUniform("depthCameraImage", 1);
    mFullscreenWithDepthShader->setUniform("znear", znear);
    mFullscreenWithDepthShader->setUniform("zfar", zfar);
    mFullscreenWithDepthShader->setUniform("depthScale", depthScale);
}

Renderer::~Renderer()
{
}

void Renderer::setTextures(GLuint colourTextures[2], GLuint depthTextures[2])
{
    mColourTextures[0] = colourTextures[0];
    mColourTextures[1] = colourTextures[1];
    mDepthTextures[0] = depthTextures[0];
    mDepthTextures[1] = depthTextures[1];
}

void Renderer::setExtrinsics(glm::mat4 extrToEye[2])
{
    mExtrToEye[0] = extrToEye[0];
    mExtrToEye[1] = extrToEye[1];
}

void Renderer::setViewport(cv::Point pos, cv::Size size)
{
    glViewport(pos.x, pos.y, size.width, size.height);
}

void Renderer::renderScene(int eye)
{
    glm::mat4 thisView = mExtrToEye[eye] * mView;

    if (mState == RS_COLOUR)
    {
        // Render captured frame from the cameras
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, mColourTextures[eye]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, mDepthTextures[eye]);
        mFullscreenWithDepthShader->bind();
        mQuad->render();

        // Render "other" virtual objects
        // ...

        // Overlay stuff
        if (mAlignmentState == AS_SEARCHING) // Render "guiding" model
        {
            mTargetEntity->render(thisView, mProjection);
        }
        else if (mAlignmentState == AS_FOUND) // Render overlays
        {
            // Disable colour and depth writing, and enable writing to the stencil buffer
            glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
            glDepthMask(GL_FALSE);
            glStencilMask(0xFF);

            // Stencil test always passes, and set ref=1
            glStencilFunc(GL_ALWAYS, 1, 0xFF);

            // Replace stencil value with 'ref' when the stencil test passes i.e. always, and when the
            // depth test also passes
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
            glClear(GL_STENCIL_BUFFER_BIT);
            mExpandedTargetEntity->render(thisView, mProjection);

            // Enable colour and depth writing, and disable writing to the stencil buffer
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glDepthMask(GL_TRUE);
            glStencilMask(0x0); // This also ignores glStencilOp as stencil buffer is write-protected

            // Clear the depth buffer
            glClear(GL_DEPTH_BUFFER_BIT);

            // Draw only when ref == stencil i.e. when a 1 is written to the stencil buffer
            glStencilFunc(GL_EQUAL, 1, 0xFF);

            // Render overlay
            mOverlay->render(thisView, mProjection);

            // Disable stencil function
            glStencilFunc(GL_ALWAYS, 1, 0xFF);

            // Clear the depth buffer
            glClear(GL_DEPTH_BUFFER_BIT);

            // Draw the model as a "frame"
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            mTargetEntity->getShader()->setUniform("diffuseColour", glm::vec4(0.4f, 0.8f, 0.4f, 0.2f));
            mTargetEntity->render(thisView, mProjection);
            mTargetEntity->getShader()->setUniform("diffuseColour", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
            glDisable(GL_BLEND);
        }
    }
    else if (mState == RS_DEBUG_DEPTH)
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, mDepthTextures[eye]);
        mFullscreenShader->bind();
        mQuad->render();
    }
    else if (mState == RS_DEBUG_KFUSION)
    {
        mTracker->getCurrentView(mTrackerDebug.getCVMat());
        mTrackerDebug.update();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, mTrackerDebug.getGLTexture());
        mFullscreenShader->bind();
        mQuad->render();
    }
}

void Renderer::setState(RendererState rs)
{
    mState = rs;
}

void Renderer::beginSearchingFor(unique_ptr<Entity> entity)
{
    mAlignmentState = AS_SEARCHING;
    mTargetEntity = std::move(entity);
    mTargetEntity->getShader()->setUniform("diffuseColour", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    mExpandedTargetEntity = make_unique<Entity>(mTargetEntity->getModel(), mTargetEntity->getShader());
    mOverlay = make_unique<Entity>(
        make_shared<Model>("../media/meshes/graymatter.stl"),
        mTargetEntity->getShader());
}

void Renderer::setObjectFound(glm::mat4 transform)
{
    mAlignmentState = AS_FOUND;
    mTargetEntity->setTransform(transform);
    mExpandedTargetEntity->setTransform(transform * glm::scale(glm::mat4(), glm::vec3(1.1f, 1.1f, 1.1f)));
    mOverlay->setTransform(transform);
}

void Renderer::setViewMatrix(const glm::mat4& view)
{
    mView = view;
}

cv::Size Renderer::getBackbufferSize() const
{
    return mBackbufferSize;
}

float Renderer::getZNear() const
{
    return mZNear;
}

float Renderer::getZFar() const
{
    return mZFar;
}

Entity* Renderer::getTargetEntity()
{
    return mTargetEntity.get();
}
