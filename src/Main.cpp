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

    virtual void render() = 0;
    virtual void keyEvent(int key, int scancode, int action, int mods) = 0;

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
        mZedCamera = new ZEDCamera();
        mRSCamera = new F200CameraColour(640, 480, 60);

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

        // Find chessboard corners
        cv::Mat greyscale;
        cvtColor(mFrame[0], greyscale, cv::COLOR_BGR2GRAY);
        cv::Size boardSize(9, 6);
        static std::vector<cv::Point2f> corners;
        bool valid = findChessboardCorners(greyscale, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
        if (valid)
            cout << "Found some chessboard corners!" << endl;
        drawChessboardCorners(mFrame[0], boardSize, cv::Mat(corners), valid);

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
    }

private:
    CameraSource* mZedCamera;
    CameraSource* mRSCamera;

    GLuint mTexture[2];
    cv::Mat mFrame[2];
};

// Rift Viewer
class RiftView : public App
{
public:
    RiftView() :
        mShowRealsense(false)
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

        mZedCamera = new ZEDCamera;
        mRSCamera = new F200CameraDepth(640, 480, 60);
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

        CameraSource* source = mShowRealsense ? (CameraSource*)mRSCamera : (CameraSource*)mZedCamera;

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
            mShowRealsense = !mShowRealsense;
    }

private:
#ifdef USE_OCULUS
    ovrSession mSession;
    ovrGraphicsLuid mLuid;
#endif

    CameraSource* mZedCamera;
    CameraSource* mRSCamera;
    bool mShowRealsense;

};

// Application entry point
int main(int argc, char** argv)
{
    try
    {
        if (!glfwInit())
            return EXIT_FAILURE;

        // Set up the window
        GLFWwindow* window = glfwCreateWindow(1280, 480, "RiftAR", nullptr, nullptr);
        if (!window)
            THROW_ERROR("Failed to create a window");
        glfwMakeContextCurrent(window);

        // Set up GL
        if (gl3wInit())
            THROW_ERROR("Failed to load GL3W");

        // Set up the application
        App* app = new CalibrateCameras;

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