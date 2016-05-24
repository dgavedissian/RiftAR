#include "lib/Common.h"
#include "lib/Model.h"

#include "RenderContext.h"

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
