#include "Common.h"
#include "F200Camera.h"         // RealsenseF200
#include "ZEDCamera.h"          // ZED SDK
#include "ZEDOpenCVCamera.h"    // OpenCV camera

#include "CameraSource.h"
#include "Rectangle2D.h"
#include "Shader.h"

#include <OVR_CAPI.h>
#include <OVR_CAPI_GL.h>

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
        static Shader shader("../media/quad.vs", "../media/quad.fs");
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
        mFrameIndex(0)
    {
    }

    void init() override
    {
        ovrResult result = ovr_Initialize(nullptr);
        if (OVR_FAILURE(result))
            THROW_ERROR("Failed to initialise LibOVR");

        // Create a context for the rift device
        result = ovr_Create(&mSession, &mLuid);
        if (OVR_FAILURE(result))
            THROW_ERROR("Oculus Rift not detected");

        // Set up the cameras
        mZedCamera = new ZEDCamera();
        mRSCamera = new F200Camera(640, 480, 60, F200Camera::DEPTH);
        mRSCamera->setStream(F200Camera::DEPTH);

        // Get the texture sizes of Oculus eyes
        mHmdDesc = ovr_GetHmdDesc(mSession);
        ovrSizei textureSize0 = ovr_GetFovTextureSize(mSession, ovrEye_Left, mHmdDesc.DefaultEyeFov[0], 1.0f);
        ovrSizei textureSize1 = ovr_GetFovTextureSize(mSession, ovrEye_Right, mHmdDesc.DefaultEyeFov[1], 1.0f);

        // Compute the final size of the render buffer
        mBufferSize.w = textureSize0.w + textureSize1.w;
        mBufferSize.h = std::max(textureSize0.h, textureSize1.h);

        // Initialize OpenGL swap textures to render
        ovrTextureSwapChainDesc descTextureSwap = {};
        descTextureSwap.Type = ovrTexture_2D;
        descTextureSwap.ArraySize = 1;
        descTextureSwap.Width = mBufferSize.w;
        descTextureSwap.Height = mBufferSize.h;
        descTextureSwap.MipLevels = 1;
        descTextureSwap.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
        descTextureSwap.SampleCount = 1;
        descTextureSwap.StaticImage = ovrFalse;

        // Create the OpenGL texture swap chain, and enable linear filtering on each texture in the swap chain
        result = ovr_CreateTextureSwapChainGL(mSession, &descTextureSwap, &mTextureChain);
        int length = 0;
        ovr_GetTextureSwapChainLength(mSession, mTextureChain, &length);
        if (OVR_SUCCESS(result))
        {
            for (int i = 0; i < length; ++i)
            {
                GLuint chainTexId;
                ovr_GetTextureSwapChainBufferGL(mSession, mTextureChain, i, &chainTexId);
                glBindTexture(GL_TEXTURE_2D, chainTexId);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            }
        }

        // Generate frame buffer to render
        glGenFramebuffers(1, &mFramebufferId);

        // Generate depth buffer of the frame buffer
        glGenTextures(1, &mDepthBufferId);
        glBindTexture(GL_TEXTURE_2D, mDepthBufferId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, mBufferSize.w, mBufferSize.h, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);

        // Create a mirror texture which is used to display the result in the GLFW window
        ovrMirrorTextureDesc descMirrorTexture;
        memset(&descMirrorTexture, 0, sizeof(descMirrorTexture));
        descMirrorTexture.Width = getSize().width;
        descMirrorTexture.Height = getSize().height;
        descMirrorTexture.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
        result = ovr_CreateMirrorTextureGL(mSession, &descMirrorTexture, &mMirrorTexture);
        if (!OVR_SUCCESS(result))
            THROW_ERROR("Failed to create the mirror texture");
        ovr_GetMirrorTextureBufferGL(mSession, mMirrorTexture, &mMirrorTextureId);

        // Create rendering primitives
        mQuad = new Rectangle2D(glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 1.0f));
        mQuadShader = new Shader("../media/quad.vs", "../media/quad_inv.fs");
    }

    ~RiftView()
    {
        delete mQuad;
        delete mQuadShader;

        delete mZedCamera;
        delete mRSCamera;

        ovr_Destroy(mSession);
        ovr_Shutdown();
    }

    void render() override
    {
        // Get texture swap index where we must draw our frame
        GLuint curTexId;
        int curIndex;
        ovr_GetTextureSwapChainCurrentIndex(mSession, mTextureChain, &curIndex);
        ovr_GetTextureSwapChainBufferGL(mSession, mTextureChain, curIndex, &curTexId);

        // Call ovr_GetRenderDesc each frame to get the ovrEyeRenderDesc, as the returned values (e.g. HmdToEyeOffset) may change at runtime.
        ovrEyeRenderDesc eyeRenderDesc[2];
        eyeRenderDesc[0] = ovr_GetRenderDesc(mSession, ovrEye_Left, mHmdDesc.DefaultEyeFov[0]);
        eyeRenderDesc[1] = ovr_GetRenderDesc(mSession, ovrEye_Right, mHmdDesc.DefaultEyeFov[1]);
        ovrVector3f hmdToEyeOffset[2];
        hmdToEyeOffset[0] = eyeRenderDesc[0].HmdToEyeOffset;
        hmdToEyeOffset[1] = eyeRenderDesc[1].HmdToEyeOffset;

        // Get eye poses, feeding in correct IPD offset
        ovrPosef eyeRenderPose[2];
        double sensorSampleTime;
        ovr_GetEyePoses(mSession, mFrameIndex, ovrTrue, hmdToEyeOffset, eyeRenderPose, &sensorSampleTime);

        // Update the textures
        mZedCamera->capture();
        mZedCamera->updateTextures();

        // Bind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, mFramebufferId);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, curTexId, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, mDepthBufferId, 0);

        // Clear the frame buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0, 0, 0, 1);

        // Render for each Oculus eye the equivalent ZED image
        mQuadShader->bind();
        for (int eye = 0; eye < 2; eye++)
        {
            // Set the left or right vertical half of the buffer as the viewport
            glViewport(eye == ovrEye_Left ? 0 : mBufferSize.w / 2, 0, mBufferSize.w / 2, mBufferSize.h);

            // Bind the left or right ZED image
            glBindTexture(GL_TEXTURE_2D, mZedCamera->getTexture(eye == ovrEye_Left ? CameraSource::LEFT : CameraSource::RIGHT));
            mQuad->render();
        }

        /*
        // Avoids an error when calling SetAndClearRenderSurface during next iteration.
        // Without this, during the next while loop iteration SetAndClearRenderSurface
        // would bind a framebuffer with an invalid COLOR_ATTACHMENT0 because the texture ID
        // associated with COLOR_ATTACHMENT0 had been unlocked by calling wglDXUnlockObjectsNV.
        glBindFramebuffer(GL_FRAMEBUFFER, mFramebufferId);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
        */

        // Commit changes to the textures so they get picked up frame
        ovr_CommitTextureSwapChain(mSession, mTextureChain);

        // Draw the mirror texture
        /*
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, getSize().width, getSize().height);
        glBindTexture(GL_TEXTURE_2D, mMirrorTextureId);
        mQuad->render();
        */

        // A frame has been completed
        mFrameIndex++;
    }

    void keyEvent(int key, int scancode, int action, int mods) override
    {
    }

    cv::Size getSize() override
    {
        return cv::Size(1280, 720);
    }

private:
    ovrSession mSession;
    ovrGraphicsLuid mLuid;
    ovrHmdDesc mHmdDesc;
    ovrSizei mBufferSize;
    ovrTextureSwapChain mTextureChain;
    GLuint mFramebufferId;
    GLuint mDepthBufferId;
    ovrMirrorTexture mMirrorTexture;
    GLuint mMirrorTextureId;

    Rectangle2D* mQuad;
    Shader* mQuadShader;

    ZEDCamera* mZedCamera;
    F200Camera* mRSCamera;

    int mFrameIndex;

};

// Application entry point
int main(int argc, char** argv)
{
    try
    {
        if (!glfwInit())
            return EXIT_FAILURE;

        // Set up the application
        App* app = new RiftView;

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