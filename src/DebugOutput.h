#pragma once

#include "OutputContext.h"

#include "lib/Rectangle2D.h"
#include "lib/Shader.h"

class DebugOutput : public OutputContext
{
public:
    DebugOutput(RenderContext& ctx);
    ~DebugOutput();

    void renderScene(RenderContext& ctx) override;

    // TODO: Move this functionality to the RenderContext
    void toggleDebug() { mShowColour = !mShowColour; }

private:
    bool mShowColour;

    Rectangle2D* mQuad;
    Shader* mFullscreenShader;
    Shader* mFullscreenWithDepthShader;

};