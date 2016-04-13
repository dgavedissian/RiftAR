#pragma once

#include <pxcsensemanager.h>

// RealSense F200 interface
class RealsenseF200
{
public:
	RealsenseF200(int width, int height, float frameRate);
	~RealsenseF200();

    void bindAndUpdate();

    GLuint getTexture() const { return mTextureID; }

private:
	PXCSenseManager *mSenseManager;
	PXCRangeF32 mFrameRate;
	PXCSizeI32 mFrameSize;

	GLuint mTextureID;
};