#include <iostream>
#include "CameraRSF200.h"

#if 0

CameraRSF200::CameraRSF200()
	: m_frame_ready(false)
{
	sense_manager = PXCSenseManager::CreateInstance();
	if (!sense_manager)
	{
		wprintf_s(L"Unable to create the SenseManager\n");
	}
}

CameraRSF200::~CameraRSF200()
{
	if (sense_manager)
	{
		proj->Release();
		sense_manager->Close();
		sense_manager->Release();
	}
}

bool CameraRSF200::ready()
{
	return m_frame_ready;
}

void CameraRSF200::init(int frameWidth, int frameHeight, float frameRate)
{
	if (sense_manager)
	{
		m_frame_rate.min = frameRate;
		m_frame_rate.max = frameRate;

		m_frame_size.width = frameWidth;
		m_frame_size.height = frameHeight;

		cv::Size2i captureSize = cv::Size2i(m_frame_size.width, m_frame_size.height);
		m_color_frame = cv::Mat(captureSize, CV_8UC3);
		m_depth_frame = cv::Mat(captureSize, CV_16UC1);
		m_infra_frame = cv::Mat(captureSize, CV_8UC1);
		m_warped_color_frame = cv::Mat(captureSize, CV_8UC3);
		m_warped_depth_frame = cv::Mat(captureSize, CV_16UC1);

		// Select the color and depth streams
		PXCVideoModule::StreamDesc sdesc;
		sdesc.frameRate = m_frame_rate;
		sdesc.sizeMin = m_frame_size;
		sdesc.sizeMax = m_frame_size;

		PXCVideoModule::DataDesc ddesc = {};
		ddesc.deviceInfo.streams = PXCCapture::STREAM_TYPE_COLOR | PXCCapture::STREAM_TYPE_DEPTH | PXCCapture::STREAM_TYPE_IR;
		ddesc.streams.color = sdesc;
		ddesc.streams.depth = sdesc;
		ddesc.streams.ir = sdesc;
		sense_manager->EnableStreams(&ddesc);

		// Initializes
		auto sts = sense_manager->Init(this);

		// Stream Data
		sense_manager->StreamFrames(false);

		proj = sense_manager->QueryCaptureManager()->QueryDevice()->CreateProjection();

		if (sts < PXC_STATUS_NO_ERROR)
		{
			wprintf_s(L"Failed to locate any video stream(s)\n");
		}
	}
}

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

pxcStatus PXCAPI CameraRSF200::OnNewSample(pxcUID, PXCCapture::Sample *sample)
{
	m_frame_ready = false;
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

	m_frame_ready = true;

	// return NO_ERROR to continue, or any error to exit the loop
	return PXC_STATUS_NO_ERROR;
}

#endif
