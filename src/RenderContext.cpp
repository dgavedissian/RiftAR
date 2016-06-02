#include "Common.h"
#include "RenderContext.h"

#include "lib/Model.h"
#include "lib/Shader.h"
#include "lib/Rectangle2D.h"

RenderContext::RenderContext(bool invertColour, float znear, float zfar, float depthScale) :
    mShowColour(true),
    mZNear(znear),
    mZFar(zfar)
{
    // Create rendering primitives
    mQuad = new Rectangle2D(glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 1.0f));
    mFullscreenShader = new Shader("../media/quad.vs", "../media/quad.fs");
    mFullscreenShader->bind();
    mFullscreenShader->setUniform("invertColour", invertColour);

    // Create objects
    mFullscreenWithDepthShader = new Shader("../media/quad.vs", "../media/quad_depth.fs");
    mFullscreenWithDepthShader->bind();
    mFullscreenWithDepthShader->setUniform("invertColour", invertColour);
    mFullscreenWithDepthShader->setUniform("rgbCameraImage", 0);
    mFullscreenWithDepthShader->setUniform("depthCameraImage", 1);
    mFullscreenWithDepthShader->setUniform("znear", znear);
    mFullscreenWithDepthShader->setUniform("zfar", zfar);
    mFullscreenWithDepthShader->setUniform("depthScale", depthScale);
}

RenderContext::~RenderContext()
{
    delete mQuad;
    delete mFullscreenShader;
    delete mFullscreenWithDepthShader;
}

void RenderContext::setViewport(cv::Point pos, cv::Size size)
{
    glViewport(pos.x, pos.y, size.width, size.height);
}

void RenderContext::renderScene(int eye)
{
    glm::mat4 thisView = eyeMatrix[eye] * view;

    if (mShowColour)
    {
        // Render captured frame from the cameras
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colourTextures[eye]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthTextures[eye]);
        mFullscreenWithDepthShader->bind();
        mQuad->render();

        // Render "other" virtual objects
        // ...

        // Overlay stuff
        if (lookingForHead) // Render "guiding" model
        {
            alignmentModel->render(thisView, projection);
        }
        else if (foundTransform) // Render overlays
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
            expandedAlignmentModel->render(thisView, projection);

            // Enable colour and depth writing, and disable writing to the stencil buffer
            glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
            glDepthMask(GL_TRUE);
            glStencilMask(0x0); // This also ignores glStencilOp as stencil buffer is write-protected

            // Clear the depth buffer
            glClear(GL_DEPTH_BUFFER_BIT);

            // Draw only when ref == stencil i.e. when a 1 is written to the stencil buffer
            glStencilFunc(GL_EQUAL, 1, 0xFF);

            // Render overlay
            model->render(thisView, projection);

            // Disable stencil function
            glStencilFunc(GL_ALWAYS, 1, 0xFF);
        }
    }
    else
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depthTextures[eye]);
        mFullscreenShader->bind();
        mQuad->render();
    }
}
