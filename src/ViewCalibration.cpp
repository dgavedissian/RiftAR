#include "Common.h"
#include "F200Camera.h"
#include "ZEDCamera.h"

class ViewCalibration : public App
{
public:
    ViewCalibration() :
        mToggle(false)
    {
    }

    void init() override
    {
        mZedCamera = new ZEDCamera();
        mRSCamera = new F200Camera(640, 480, 60, F200Camera::COLOUR);
        mRSCamera->setStream(F200Camera::COLOUR);
    }

    ~ViewCalibration()
    {
        delete mZedCamera;
        delete mRSCamera;
    }

    void render() override
    {
        // Read the left frame from both cameras
        mZedCamera->capture();
        mZedCamera->copyFrameIntoCVImage(CameraSource::LEFT, &mFrame[0]);
        mRSCamera->capture();
        mRSCamera->copyFrameIntoCVImage(CameraSource::LEFT, &mFrame[1]);

        // Stereo Calibration

        // Display them
        //cv::imshow("View", mFrame[mToggle ? 0 : 1]);
        //cv::imshow("View", mToggle ? mFrame[0] : undistorted);
    }

    void keyEvent(int key, int scancode, int action, int mods) override
    {
        if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
            mToggle = !mToggle;
    }

    cv::Size getSize() override
    {
        return cv::Size(300, 300);
    }

private:
    bool mToggle;

    ZEDCamera* mZedCamera;
    F200Camera* mRSCamera;

    cv::Mat mFrame[2];
};

DEFINE_MAIN(ViewCalibration)
