#include "Common.h"
#include "F200Camera.h"         // RealsenseF200
#include "ZEDCamera.h"          // ZED
#include <OVR_CAPI.h>           // LibOVR

#include "FullscreenQuad.h"
#include "Shader.h"

void ShowError(const std::string& error)
{
    MessageBoxA(0, error.c_str(), "Error", MB_ICONERROR);
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
            ShowError("Failed to create a window");
            glfwTerminate();
            return EXIT_FAILURE;
        }
        glfwMakeContextCurrent(window);

        // Set up GL
        if (gl3wInit())
        {
            ShowError("Failed to initialise GL3W");
            glfwTerminate();
            return EXIT_FAILURE;
        }

        // Set up OVR
        ovrResult result = ovr_Initialize(nullptr);
        if (OVR_FAILURE(result))
        {
            ShowError("Failed to initialise LibOVR");
            glfwTerminate();
            return EXIT_FAILURE;
        }

        // Create a context for the rift device
        ovrSession session;
        ovrGraphicsLuid luid;
        result = ovr_Create(&session, &luid);
        if (OVR_FAILURE(result))
        {
            ShowError("Oculus Rift not detected");
            ovr_Shutdown();
            glfwTerminate();
            return EXIT_FAILURE;
        }

        // Set up the cameras
        ZEDCamera zedCamera;
        F200Camera f200Camera(640, 480, 60);

        // Create the fullscreen quad and shader
        FullscreenQuad quad;
        Shader shader("../media/fullscreenquad.vs", "../media/fullscreenquad.fs");

        // Main loop
        bool showRealsense = false;
        while (!glfwWindowShouldClose(window))
        {
            glClear(GL_COLOR_BUFFER_BIT);

            // Render
            if (showRealsense)
            {
                f200Camera.bindAndUpdate();
                glBindTextureUnit(0, zedCamera.getTexture());
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
        ShowError(e.what());
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