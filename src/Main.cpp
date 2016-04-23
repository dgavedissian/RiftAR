#include "Common.h"
#include "F200Camera.h"         // RealsenseF200
#include "ZEDCamera.h"          // ZED SDK
#include "CVCamera.h"           // OpenCV camera
#include <OVR_CAPI.h>           // LibOVR

#include "Rectangle2D.h"
#include "Shader.h"

void showError(const std::string& error)
{
    MessageBoxA(0, error.c_str(), "Error", MB_ICONERROR);
}

bool showRealsense = false;

void keyFunc(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        showRealsense = !showRealsense;
}

// Application entry point
int main(int argc, char** argv)
{
    try
    {
        if (!glfwInit())
            return EXIT_FAILURE;

        // Set up the window
        GLFWwindow* window = glfwCreateWindow(640, 480, "Hello World", nullptr, nullptr);
        if (!window)
        {
            showError("Failed to create a window");
            glfwTerminate();
            return EXIT_FAILURE;
        }
        glfwMakeContextCurrent(window);

        // Set up GL
        if (gl3wInit())
        {
            showError("Failed to initialise GL3W");
            glfwTerminate();
            return EXIT_FAILURE;
        }

        // Set up OVR
        ovrResult result = ovr_Initialize(nullptr);
        if (OVR_FAILURE(result))
        {
            showError("Failed to initialise LibOVR");
            glfwTerminate();
            return EXIT_FAILURE;
        }

        // Create a context for the rift device
        ovrSession session;
        ovrGraphicsLuid luid;
        result = ovr_Create(&session, &luid);
        if (OVR_FAILURE(result))
        {
            showError("Oculus Rift not detected");
            ovr_Shutdown();
            glfwTerminate();
            return EXIT_FAILURE;
        }

        // Set up the cameras
        CVCamera zedCamera(4);
        F200Camera f200Camera(640, 480, 60);

        // Create the fullscreen quad and shader
        Rectangle2D quad(glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 1.0f));
        Shader shader("../media/fullscreenquad.vs", "../media/fullscreenquad.fs");

        // Main loop
        glfwSetKeyCallback(window, keyFunc);
        while (!glfwWindowShouldClose(window))
        {
            glClear(GL_COLOR_BUFFER_BIT);

            // Render
            if (showRealsense)
            {
                f200Camera.bindAndUpdate();
                glBindTextureUnit(0, f200Camera.getTexture());
            }
            else
            {
                zedCamera.bindAndUpdate();
                glBindTextureUnit(0, zedCamera.getTexture());
            }
            shader.bind();
            quad.render();

            // Swap buffers
            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        ovr_Destroy(session);
        ovr_Shutdown();
        glfwTerminate();
        return EXIT_SUCCESS;
    }
    catch (std::runtime_error& e)
    {
        showError(e.what());
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