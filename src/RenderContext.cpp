#include "lib/Common.h"
#include "lib/Model.h"

#include "RenderContext.h"

void RenderContext::renderScene(int eye)
{
    model->render(eyeMatrix[eye] * view, projection);
}
