#include "Common.h"
#include "RealsenseF200.h"      // Realsense Camera
#include <OVR_CAPI.h>           // LibOVR
//#include <zed/Camera.hpp>       // ZED SDK

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
            return -1;

        // Set up the window
        GLFWwindow* window = glfwCreateWindow(640, 480, "Hello World", nullptr, nullptr);
        if (!window)
        {
            ShowError("Failed to create a window");
            glfwTerminate();
            return EXIT_FAILURE;
        }
        glfwMakeContextCurrent(window);

        // Set up GL3W and load OpenGL extensions
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

        // Set up the ZED
        /*
        sl::zed::Camera* zed = new sl::zed::Camera(sl::zed::HD720);
        int zWidth = zed->getImageSize().width;
        int zHeight = zed->getImageSize().height;
        sl::zed::ERRCODE zederror = zed->init(sl::zed::MODE::PERFORMANCE, 0);
        if (zederror != sl::zed::SUCCESS)
        {
        ShowError("ZED camera not detected");
        ovr_Shutdown();
        glfwTerminate();
        return EXIT_FAILURE;
        }

        // Generate OpenGL textures for the left and right eyes of the ZED camera
        GLuint zedTextureL, zedTextureR;
        glGenTextures(1, &zedTextureL);
        glBindTexture(GL_TEXTURE_2D, zedTextureL);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, zWidth, zHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
        */

        // Set up the Realsense
        RealsenseF200 camera(640, 480, 60);

        FullscreenQuad quad;
        Shader shader("../media/fullscreenquad.vs", "../media/fullscreenquad.fs");

        // Main loop
        while (!glfwWindowShouldClose(window))
        {
            glClear(GL_COLOR_BUFFER_BIT);

            // Render
            camera.bindAndUpdate();
            glBindTextureUnit(0, camera.getTexture());
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