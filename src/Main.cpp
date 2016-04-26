#include "Common.h"
#include "F200Camera.h"         // RealsenseF200
#include "ZEDCamera.h"          // ZED SDK
#include "ZEDOpenCVCamera.h"    // OpenCV camera
#include "RiftPipeline.h"

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
        GLFWwindow* window = glfwCreateWindow(1280, 480, "RiftAR", nullptr, nullptr);
        if (!window)
            throw std::runtime_error("Failed to create a window");
        glfwMakeContextCurrent(window);

        // Set up GL
        if (gl3wInit())
            throw std::runtime_error("Failed to load GL3W");

        // Set up OVR
        RiftPipeline pipeline;

        // Set up the cameras
        ZEDCamera zedCamera;
        //ZEDOpenCVCamera zedCamera(1);
        F200Camera rsCamera(640, 480, 60, true);

        // Main loop
        glfwSetKeyCallback(window, keyFunc);
        while (!glfwWindowShouldClose(window))
        {
            glClear(GL_COLOR_BUFFER_BIT);

            // Render
            if (showRealsense)
            {
                pipeline.display(&rsCamera);
            }
            else
            {
                pipeline.display(&zedCamera);
            }

            // Swap buffers
            glfwSwapBuffers(window);
            glfwPollEvents();
        }

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