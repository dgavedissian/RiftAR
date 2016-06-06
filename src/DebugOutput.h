#pragma once

#include "lib/Rectangle2D.h"
#include "lib/Shader.h"

#include "OutputContext.h"

class DebugOutput : public OutputContext
{
public:
    void renderScene(Renderer* ctx) override;

};