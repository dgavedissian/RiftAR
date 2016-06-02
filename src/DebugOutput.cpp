#include "Common.h"
#include "DebugOutput.h"

void DebugOutput::renderScene(RenderContext* ctx)
{
    for (int i = 0; i < 2; i++)
    {
        ctx->setViewport(
            cv::Point(i == 0 ? 0 : ctx->backbufferSize.width / 2, 0),
            cv::Size(ctx->backbufferSize.width / 2, ctx->backbufferSize.height));
        ctx->renderScene(i);
    }
}
