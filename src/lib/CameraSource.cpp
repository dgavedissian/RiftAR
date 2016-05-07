#include "Common.h"
#include "CameraSource.h"

uint CameraSource::getWidth(uint camera) const
{
    return getIntrinsics(camera).width;
}

uint CameraSource::getHeight(uint camera) const
{
    return getIntrinsics(camera).height;
}
