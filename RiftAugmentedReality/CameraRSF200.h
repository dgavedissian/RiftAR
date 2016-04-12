#pragma once

#include <pxcsensemanager.h>
#include <pxcprojection.h>

// RealSense F200 interface
class CameraRSF200 : public PXCSenseManager::Handler
{
public:
	CameraRSF200();
	~CameraRSF200();

	void init(int frameWidth, int frameHeight, float frameRate);

	bool ready();

		/*
	bool getAllFrames(cv::Mat &color, cv::Mat &depth, cv::Mat &infra);

	bool getColorFrame(cv::Mat &color);

	bool getDepthFrame(cv::Mat &depth);

	bool getInfraFrame(cv::Mat &infra);

	bool getColorDepthFrames(cv::Mat &color, cv::Mat &depth);

	bool getColorAndAlignedDepthFrames(cv::Mat &color, cv::Mat &depth);

	bool getDepthAndAlignedColorFrames(cv::Mat &color, cv::Mat &depth);

	bool getAlignedColorFrame(cv::Mat &color);

	bool getAlignedDepthFrame(cv::Mat &depth);
	*/

	/*double getFPS()
	{
		return fps.getFrequency();
	}*/

	pxcStatus PXCAPI OnNewSample(pxcUID, PXCCapture::Sample *sample) override;

private:
	bool mFrameReady;
	PXCSenseManager *mSenseManager;
	PXCProjection *proj;
	PXCRangeF32 mFrameRate;
	PXCSizeI32 mFrameSize;

	GLuint mTextureID;
	/*
	cv::Mat m_color_frame;
	cv::Mat m_depth_frame;
	cv::Mat m_infra_frame;
	cv::Mat m_warped_color_frame;
	cv::Mat m_warped_depth_frame;
	FrequencyMonitor fps;
	*/
};