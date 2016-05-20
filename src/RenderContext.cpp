#include "lib/Common.h"
#include "lib/Model.h"

#include "RenderContext.h"

void RenderContext::renderScene(int eye)
{
    if (foundTransform)
        glDepthFunc(GL_ALWAYS);
    model->render(eyeMatrix[eye] * view, projection);
    if (foundTransform)
        glDepthFunc(GL_LESS);
}
