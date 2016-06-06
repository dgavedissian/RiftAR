#pragma once

#include "Renderer.h"

class OutputContext
{
public:
    virtual ~OutputContext() {}
    virtual void renderScene(Renderer* ctx) = 0;
};
