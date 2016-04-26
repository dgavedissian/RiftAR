#pragma once

#include <OVR_CAPI.h>

class CameraSource;

class RiftPipeline
{
public:
    RiftPipeline();
    ~RiftPipeline();

    void display(CameraSource* source);

private:
    ovrSession mSession;
    ovrGraphicsLuid mLuid;
};