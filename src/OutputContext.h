#pragma once

#include "RenderContext.h"

class OutputContext
{
public:
    virtual ~OutputContext() {}
    virtual void renderScene(RenderContext& ctx) = 0;
};
