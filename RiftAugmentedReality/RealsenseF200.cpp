#include "Common.h"
#include "RealsenseF200.h"

RealsenseF200::RealsenseF200(int width, int height, float frameRate)
{
	mSenseManager = PXCSenseManager::CreateInstance();

    mFrameRate.min = frameRate;
    mFrameRate.max = frameRate;

    mFrameSize.width = width;
    mFrameSize.height = height;

    // Select the color and depth streams
    PXCVideoModule::StreamDesc sdesc;
    sdesc.frameRate = mFrameRate;
    sdesc.sizeMin = mFrameSize;
    sdesc.sizeMax = mFrameSize;

    PXCVideoModule::DataDesc ddesc = {};
    ddesc.deviceInfo.streams = PXCCapture::STREAM_TYPE_COLOR | PXCCapture::STREAM_TYPE_DEPTH;
    ddesc.streams.color = sdesc;
    ddesc.streams.depth = sdesc;
    mSenseManager->EnableStreams(&ddesc);

    // Initializes
    pxcStatus status = mSenseManager->Init();
    cout << status << endl;
    mSenseManager->Init();
    if (status < PXC_STATUS_NO_ERROR)
    {
        throw std::runtime_error("Failed to locate any video stream(s)");
    }

    // Set up image
    glGenTextures(1, &mTextureID);
    glBindTexture(GL_TEXTURE_2D, mTextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

RealsenseF200::~RealsenseF200()
{
	if (mSenseManager)
	{
		mSenseManager->Close();
		mSenseManager->Release();
	}
}

void RealsenseF200::bindAndUpdate()
{
    if (mSenseManager->AcquireFrame(true) < PXC_STATUS_NO_ERROR)
        return;
    
    glBindTexture(GL_TEXTURE_2D, mTextureID);

    PXCCapture::Sample* sample = mSenseManager->QuerySample();

    // Grab data
    // TODO: Use a pixel buffer object instead of glTexSubImage2D
    PXCImage::ImageData imageData;
    sample->color->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_RGB24, &imageData);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mFrameSize.width, mFrameSize.height, GL_RGB, GL_UNSIGNED_BYTE, reinterpret_cast<void*>(imageData.planes[0]));
    sample->color->ReleaseAccess(&imageData);

    mSenseManager->ReleaseFrame();
}