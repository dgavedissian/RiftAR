#include "Common.h"
#include "StereoCamera.h"

StereoCamera::StereoCamera()
{
}

StereoCamera::~StereoCamera()
{
}

GLuint StereoCamera::getTexture(Eye e)
{
    return mTexture[e];
}

