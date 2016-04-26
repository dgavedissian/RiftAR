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

// Rift Viewer
class RiftView : public App
{
public:
    RiftView() :
        showRealsense(false)
    {
#ifdef USE_OCULUS
        ovrResult result = ovr_Initialize(nullptr);
        if (OVR_FAILURE(result))
            throw std::runtime_error("Failed to initialise LibOVR");

        // Create a context for the rift device
        result = ovr_Create(&mSession, &mLuid);
        if (OVR_FAILURE(result))
            throw std::runtime_error("Oculus Rift not detected");
#endif

        zedCamera = new ZEDCamera;
        rsCamera = new F200CameraColour(640, 480, 60);
    }

    ~RiftView()
    {
        delete zedCamera;
        delete rsCamera;

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

        CameraSource* source = showRealsense ? (CameraSource*)rsCamera : (CameraSource*)zedCamera;

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
            showRealsense = !showRealsense;
    }

private:
#ifdef USE_OCULUS
    ovrSession mSession;
    ovrGraphicsLuid mLuid;
#endif

    ZEDCamera* zedCamera;
    F200CameraColour* rsCamera;
    bool showRealsense;

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
            throw std::runtime_error("Failed to create a window");
        glfwMakeContextCurrent(window);

        // Set up GL
        if (gl3wInit())
            throw std::runtime_error("Failed to load GL3W");

        // Set up the application
        App* app = new RiftView;

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