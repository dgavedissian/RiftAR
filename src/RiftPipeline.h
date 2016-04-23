#pragma once

#include <OVR_CAPI.h>

class StereoCamera;

class RiftPipeline
{
public:
    RiftPipeline();
    ~RiftPipeline();

    void display(StereoCamera* input);

private:
    ovrSession mSession;
    ovrGraphicsLuid mLuid;
};