#include "lib/Common.h"
#include "lib/Model.h"
#include "DebugOutput.h"

DebugOutput::DebugOutput(RenderContext& ctx, bool invertColour) :
    mShowColour(false)
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
    mFullscreenWithDepthShader->setUniform("znear", ctx.znear);
    mFullscreenWithDepthShader->setUniform("zfar", ctx.zfar);
    mFullscreenWithDepthShader->setUniform("depthScale", ctx.depthScale);
}

DebugOutput::~DebugOutput()
{
    delete mQuad;
    delete mFullscreenShader;
    delete mFullscreenWithDepthShader;
}

void DebugOutput::renderScene(RenderContext& ctx)
{
    // Render each eye
    for (int i = 0; i < 2; i++)
    {
        glViewport(
            i == 0 ? 0 : ctx.backbufferSize.width / 2, 0,
            ctx.backbufferSize.width / 2, ctx.backbufferSize.height);
        if (mShowColour)
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, ctx.colourTextures[i]);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, ctx.depthTextures[i]);
            mFullscreenShader->bind();
            mQuad->render();

            ctx.model->render(ctx.view, ctx.projection);
        }
        else
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, ctx.depthTextures[i]);
            mFullscreenShader->bind();
            mQuad->render();
        }
    }
}
