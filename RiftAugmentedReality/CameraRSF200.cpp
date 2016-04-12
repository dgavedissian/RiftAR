#include <iostream>
#include "GLFW\glfw3.h"
#include "CameraRSF200.h"

CameraRSF200::CameraRSF200()
	: mFrameReady(false)
{
	mSenseManager = PXCSenseManager::CreateInstance();
	if (!mSenseManager)
	{
		wprintf_s(L"Unable to create the SenseManager\n");
	}

	// Set up image
	glGenTextures(1, &mTextureID);
	glBindTexture(GL_TEXTURE_2D, mTextureID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

CameraRSF200::~CameraRSF200()
{
	if (mSenseManager)
	{
		proj->Release();
		mSenseManager->Close();
		mSenseManager->Release();
	}
}

void CameraRSF200::init(int frameWidth, int frameHeight, float frameRate)
{
	if (mSenseManager)
	{
		mFrameRate.min = frameRate;
		mFrameRate.max = frameRate;

		mFrameSize.width = frameWidth;
		mFrameSize.height = frameHeight;

		/*
		cv::Size2i captureSize = cv::Size2i(m_frame_size.width, m_frame_size.height);
		m_color_frame = cv::Mat(captureSize, CV_8UC3);
		m_depth_frame = cv::Mat(captureSize, CV_16UC1);
		m_infra_frame = cv::Mat(captureSize, CV_8UC1);
		m_warped_color_frame = cv::Mat(captureSize, CV_8UC3);
		m_warped_depth_frame = cv::Mat(captureSize, CV_16UC1);
		*/

		// Select the color and depth streams
		PXCVideoModule::StreamDesc sdesc;
		sdesc.frameRate = mFrameRate;
		sdesc.sizeMin = mFrameSize;
		sdesc.sizeMax = mFrameSize;

		PXCVideoModule::DataDesc ddesc = {};
		ddesc.deviceInfo.streams = PXCCapture::STREAM_TYPE_COLOR | PXCCapture::STREAM_TYPE_DEPTH | PXCCapture::STREAM_TYPE_IR;
		ddesc.streams.color = sdesc;
		ddesc.streams.depth = sdesc;
		ddesc.streams.ir = sdesc;
		mSenseManager->EnableStreams(&ddesc);

		// Initializes
		auto sts = mSenseManager->Init(this);

		// Stream Data
		mSenseManager->StreamFrames(false);

		proj = mSenseManager->QueryCaptureManager()->QueryDevice()->CreateProjection();

		if (sts < PXC_STATUS_NO_ERROR)
		{
			wprintf_s(L"Failed to locate any video stream(s)\n");
		}
	}
}

bool CameraRSF200::ready()
{
	return mFrameReady;
}

pxcStatus PXCAPI CameraRSF200::OnNewSample(pxcUID, PXCCapture::Sample *sample)
{
	mFrameReady = false;
	/*
	cv::Size2i captureSize = cv::Size2i(m_frame_size.width, m_frame_size.height);

	PXCImage::ImageData imageData;
	cv::Mat capturedFrame;

	sample->color->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_RGB24, &imageData);
	capturedFrame = cv::Mat(captureSize, CV_8UC3, imageData.planes[0]);
	capturedFrame.copyTo(m_color_frame);
	sample->color->ReleaseAccess(&imageData);

	sample->depth->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_DEPTH, &imageData);
	capturedFrame = cv::Mat(captureSize, CV_16UC1, imageData.planes[0]);
	capturedFrame.copyTo(m_depth_frame);
	sample->depth->ReleaseAccess(&imageData);

	sample->ir->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_Y8, &imageData);
	capturedFrame = cv::Mat(captureSize, CV_8UC1, imageData.planes[0]);
	capturedFrame.copyTo(m_infra_frame);
	sample->ir->ReleaseAccess(&imageData);

	PXCImage * warpedColor = proj->CreateColorImageMappedToDepth(sample->depth, sample->color);
	warpedColor->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_RGB24, &imageData);
	capturedFrame = cv::Mat(captureSize, CV_8UC3, imageData.planes[0]);
	capturedFrame.copyTo(m_warped_color_frame);
	warpedColor->ReleaseAccess(&imageData);

	PXCImage * warpedDepth = proj->CreateDepthImageMappedToColor(sample->depth, sample->color);
	warpedDepth->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_DEPTH, &imageData);
	capturedFrame = cv::Mat(captureSize, CV_16UC1, imageData.planes[0]);
	capturedFrame.copyTo(m_warped_depth_frame);
	warpedDepth->ReleaseAccess(&imageData);

	if (fps.ding())
	{
		wprintf_s(L"Camera FPS: %f\n", fps.getFrequency());
	}
	*/
	mFrameReady = true;

	// return NO_ERROR to continue, or any error to exit the loop
	return PXC_STATUS_NO_ERROR;
}

/*
bool CameraRSF200::getAllFrames(cv::Mat &color, cv::Mat &depth, cv::Mat &infra)
{
	if (m_frame_ready)
	{
		m_color_frame.copyTo(color);
		m_depth_frame.copyTo(depth);
		m_infra_frame.copyTo(infra);
		return true;
	}
	else
		return false;
}

bool CameraRSF200::getColorFrame(cv::Mat &color)
{
	if (m_frame_ready)
	{
		m_color_frame.copyTo(color);
		return true;
	}
	else
		return false;
}

bool CameraRSF200::getDepthFrame(cv::Mat &depth)
{
	if (m_frame_ready)
	{
		m_depth_frame.copyTo(depth);
		return true;
	}
	else
		return false;
}

bool CameraRSF200::getInfraFrame(cv::Mat &infra)
{
	if (m_frame_ready)
	{
		m_infra_frame.copyTo(infra);
		return true;
	}
	else
		return false;
}

bool CameraRSF200::getColorDepthFrames(cv::Mat &color, cv::Mat &depth)
{
	if (m_frame_ready)
	{
		m_color_frame.copyTo(color);
		m_depth_frame.copyTo(depth);
		return true;
	}
	else
		return false;
}

bool CameraRSF200::getColorAndAlignedDepthFrames(cv::Mat &color, cv::Mat &depth)
{
	if (m_frame_ready)
	{
		m_color_frame.copyTo(color);
		m_warped_depth_frame.copyTo(depth);
		return true;
	}
	else
		return false;
}

bool CameraRSF200::getDepthAndAlignedColorFrames(cv::Mat &color, cv::Mat &depth)
{
	if (m_frame_ready)
	{
		m_warped_color_frame.copyTo(color);
		m_depth_frame.copyTo(depth);
		return true;
	}
	else
		return false;
}

bool CameraRSF200::getAlignedColorFrame(cv::Mat &color)
{
	if (m_frame_ready)
	{
		m_warped_color_frame.copyTo(color);
		return true;
	}
	else
		return false;
}

bool CameraRSF200::getAlignedDepthFrame(cv::Mat &depth)
{
	if (m_frame_ready)
	{
		m_warped_depth_frame.copyTo(depth);
		return true;
	}
	else
		return false;
}
*/