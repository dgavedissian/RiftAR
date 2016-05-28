#include "Common.h"
#include "RenderContext.h"

#include "lib/Model.h"

void RenderContext::renderScene(int eye)
{
    if (foundTransform)
    {
        glDepthFunc(GL_ALWAYS);
        alignmentModel->render(eyeMatrix[eye] * view, projection);
        glDepthFunc(GL_LESS);
    }
    else if (lookingForHead)
    {
        alignmentModel->render(eyeMatrix[eye] * view, projection);
    }
}
