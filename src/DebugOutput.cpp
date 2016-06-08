#include "Common.h"
#include "DebugOutput.h"

void DebugOutput::renderScene(Renderer* ctx, int hit)
{
    for (int i = 0; i < 2; i++)
    {
        int hitFactor = i == 0 ? hit : -hit;
        ctx->setViewport(
            cv::Point(i == 0 ? hitFactor : ctx->backbufferSize.width / 2 + hitFactor, 0),
            cv::Size(ctx->backbufferSize.width / 2, ctx->backbufferSize.height));
        ctx->renderScene(i);
    }
}
