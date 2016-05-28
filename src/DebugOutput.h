#pragma once

#include "lib/Rectangle2D.h"
#include "lib/Shader.h"

#include "OutputContext.h"

class DebugOutput : public OutputContext
{
public:
    DebugOutput(RenderContext& ctx, bool invertColour);
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