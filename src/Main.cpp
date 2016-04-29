#include "Common.h"
#include "F200Camera.h"         // RealsenseF200
#include "ZEDCamera.h"          // ZED SDK
#include "ZEDOpenCVCamera.h"    // OpenCV camera

#include "CameraSource.h"
#include "Rectangle2D.h"
#include "Shader.h"

//#define USE_OCULUS

#ifdef USE_OCULUS
#include <OVR_CAPI.h>
#endif

class App
{
public:
    App() { msCurrentApp = this; }
    virtual ~App() {}

    virtual void init() = 0;
    virtual void render() = 0;
    virtual void keyEvent(int key, int scancode, int action, int mods) = 0;
    virtual cv::Size getSize() = 0;

    // C style callback which forwards the event to the application
    static App* msCurrentApp;
    static void glfwKeyEvent(GLFWwindow*, int key, int scancode, int action, int mods)
    {
        msCurrentApp->keyEvent(key, scancode, action, mods);
    }
};

App* App::msCurrentApp = nullptr;

// Camera Calibration
class CalibrateCameras : public App
{
public:
    CalibrateCameras()
    {
    }

    void init() override
    {
        mZedCamera = new ZEDCamera();
        mRSCamera = new F200Camera(640, 480, 60, F200Camera::COLOUR);
        mRSCamera->setStream(F200Camera::COLOUR);

        // Create OpenGL images to visualise the calibration
        glGenTextures(2, mTexture);
        glBindTexture(GL_TEXTURE_2D, mTexture[0]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mZedCamera->getWidth(), mZedCamera->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, mTexture[1]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mRSCamera->getWidth(), mRSCamera->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    ~CalibrateCameras()
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

        // Display them
        static Rectangle2D leftQuad(glm::vec2(0.0f, 0.0f), glm::vec2(0.5f, 1.0f));
        static Rectangle2D rightQuad(glm::vec2(0.5f, 0.0f), glm::vec2(1.0f, 1.0f));
        static Shader shader("../media/fullscreenquad.vs", "../media/fullscreenquad.fs");
        shader.bind();
        glBindTexture(GL_TEXTURE_2D, mTexture[0]);
        TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mZedCamera->getWidth(), mZedCamera->getHeight(),
            GL_BGR, GL_UNSIGNED_BYTE, mFrame[0].ptr()));
        leftQuad.render();
        glBindTexture(GL_TEXTURE_2D, mTexture[1]);
        TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mRSCamera->getWidth(), mRSCamera->getHeight(),
            GL_BGR, GL_UNSIGNED_BYTE, mFrame[1].ptr()));
        rightQuad.render();
    }

    void keyEvent(int key, int scancode, int action, int mods) override
    {
        static cv::Size boardSize(9, 6);
        static double squareSize = 0.025; // 25mm

        if (action == GLFW_PRESS)
        {
            if (key == GLFW_KEY_SPACE)
            {
                // Find chessboard corners from both cameras
                cv::Mat greyscale;
                static std::vector<cv::Point2f> leftCorners, rightCorners;
                cvtColor(mFrame[0], greyscale, cv::COLOR_BGR2GRAY);
                bool leftValid = findChessboardCorners(greyscale, boardSize, leftCorners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
                cvtColor(mFrame[1], greyscale, cv::COLOR_BGR2GRAY);
                bool rightValid = findChessboardCorners(greyscale, boardSize, rightCorners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

                // If neither are valid, skip this frame
                if (!leftValid || !rightValid)
                    return;

                // Both are valid, add the pair of points
                mLeftCorners.push_back(leftCorners);
                mRightCorners.push_back(rightCorners);
                leftCorners.clear();
                rightCorners.clear();

                // Mark that a corner was added
                cout << "Captured a pair of corners - #" << mLeftCorners.size() << endl;
            }
            else if (key == GLFW_KEY_ENTER)
            {
                // We are done capturing pairs of corners, find extrinsic parameters
                std::vector<std::vector<cv::Point3f>> objectPoints;
                objectPoints.resize(mLeftCorners.size());
                for (int i = 0; i < mLeftCorners.size(); i++)
                {
                    for (int j = 0; j < boardSize.height; j++)
                        for (int k = 0; k < boardSize.width; k++)
                            objectPoints[i].push_back(cv::Point3f(j * squareSize, k * squareSize, 0.0));
                }

                cv::Mat R, T, E, F;
                double rms = stereoCalibrate(objectPoints, mLeftCorners, mRightCorners,
                    mZedCamera->getCaneraMatrix(), mZedCamera->getDistCoeffs(),
                    mRSCamera->getCaneraMatrix(), mRSCamera->getDistCoeffs(),
                    mFrame[0].size(), R, T, E, F,
                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5),
                    cv::CALIB_FIX_INTRINSIC);
                cout << "Stereo Calibration done with RMS error = " << rms << endl;
                cout << "R: " << R << endl;
                cout << "T: " << T << endl;
                cout << "E: " << E << endl;
                cout << "F: " << F << endl;

                // Save to a file
                cv::FileStorage fs("../stereo-params.xml", cv::FileStorage::WRITE);
                if (fs.isOpened())
                {
                    fs <<
                        "R" << R <<
                        "T" << T <<
                        "E" << E <<
                        "F" << F;
                    fs.release();
                }
                else
                {
                    THROW_ERROR("Unable to save calibration parameters");
                }
            }
        }
    }

    cv::Size getSize() override
    {
        return cv::Size(1280, 480);
    }

private:
    ZEDCamera* mZedCamera;
    F200Camera* mRSCamera;

    GLuint mTexture[2];
    cv::Mat mFrame[2];
    std::vector<std::vector<cv::Point2f>> mLeftCorners;
    std::vector<std::vector<cv::Point2f>> mRightCorners;
};


class ViewCalibrated : public App
{
public:
    ViewCalibrated() :
        mToggle(false)
    {
    }

    void init() override
    {
        mZedCamera = new ZEDCamera();
        mRSCamera = new F200Camera(640, 480, 60, F200Camera::COLOUR);
        mRSCamera->setStream(F200Camera::COLOUR);
    }

    ~ViewCalibrated()
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

// Rift Viewer
class RiftView : public App
{
public:
    RiftView() :
        mToggle(false)
    {
    }

    void init() override
    {
#ifdef USE_OCULUS
        ovrResult result = ovr_Initialize(nullptr);
        if (OVR_FAILURE(result))
            THROW_ERROR("Failed to initialise LibOVR");

        // Create a context for the rift device
        result = ovr_Create(&mSession, &mLuid);
        if (OVR_FAILURE(result))
            THROW_ERROR("Oculus Rift not detected");
#endif

        mZedCamera = new ZEDCamera();
        mRSCamera = new F200Camera(640, 480, 60, F200Camera::COLOUR | F200Camera::DEPTH);
        mRSCamera->setStream(F200Camera::DEPTH);
    }

    ~RiftView()
    {
        delete mZedCamera;
        delete mRSCamera;

#ifdef USE_OCULUS
        ovr_Destroy(mSession);
        ovr_Shutdown();
#endif
    }

    void render() override
    {
#ifdef USE_OCULUS
        // TODO
#else
        static Rectangle2D leftQuad(glm::vec2(0.0f, 0.0f), glm::vec2(0.5f, 1.0f));
        static Rectangle2D rightQuad(glm::vec2(0.5f, 0.0f), glm::vec2(1.0f, 1.0f));
        static Shader shader("../media/fullscreenquad.vs", "../media/fullscreenquad.fs");

        //CameraSource* source = mShowRealsense ? (CameraSource*)mRSCamera : (CameraSource*)mZedCamera;
        CameraSource* source = mRSCamera;
        mRSCamera->setStream(mToggle ? F200Camera::COLOUR : F200Camera::DEPTH);

        shader.bind();
        source->capture();
        source->updateTextures();
        glBindTextureUnit(0, source->getTexture(CameraSource::LEFT));
        leftQuad.render();
        glBindTextureUnit(0, source->getTexture(CameraSource::RIGHT));
        rightQuad.render();
#endif
    }

    void keyEvent(int key, int scancode, int action, int mods) override
    {
        if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
            mToggle = !mToggle;
    }

    cv::Size getSize() override
    {
        return cv::Size(1280, 480);
    }

private:
#ifdef USE_OCULUS
    ovrSession mSession;
    ovrGraphicsLuid mLuid;
#endif

    ZEDCamera* mZedCamera;
    F200Camera* mRSCamera;
    bool mToggle;

};

// Application entry point
int main(int argc, char** argv)
{
    try
    {
        if (!glfwInit())
            return EXIT_FAILURE;

        // Set up the application
        App* app = new CalibrateCameras;

        // Set up the window
        GLFWwindow* window = glfwCreateWindow(app->getSize().width, app->getSize().height, "RiftAR", nullptr, nullptr);
        if (!window)
            THROW_ERROR("Failed to create a window");
        glfwMakeContextCurrent(window);

        // Set up GL
        if (gl3wInit())
            THROW_ERROR("Failed to load GL3W");

        // Initialise
        app->init();

        // Main loop
        glfwSetKeyCallback(window, App::glfwKeyEvent);
        while (!glfwWindowShouldClose(window))
        {
            glClear(GL_COLOR_BUFFER_BIT);

            // Render
            app->render();

            // Swap buffers
            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        // Shutdown
        delete app;
        glfwTerminate();
        return EXIT_SUCCESS;
    }
    catch (std::runtime_error& e)
    {
        MessageBoxA(0, e.what(), "Error", MB_ICONERROR);
        return EXIT_FAILURE;
    }
}

// Windows subsystem entry point
// If the CONSOLE subsystem is chosen in the VS property pages, this function will just be ignored
// completely and main() will be used instead.
#ifdef _WIN32
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
    return main(__argc, __argv);
}
#endif