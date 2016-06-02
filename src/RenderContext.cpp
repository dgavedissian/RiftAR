#include "Common.h"
#include "RenderContext.h"

#include "lib/Model.h"

void RenderContext::renderScene(int eye)
{
    glm::mat4 thisView = eyeMatrix[eye] * view;

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
