#include "Common.h"

App* App::msCurrentApp = nullptr;

void App::glfwKeyEvent(GLFWwindow*, int key, int scancode, int action, int mods)
{
    msCurrentApp->keyEvent(key, scancode, action, mods);
}

int runApp(int argc, char** argv, App* app)
{
    try
    {
        if (!glfwInit())
            return EXIT_FAILURE;

        // Set up the window
        GLFWwindow* window = glfwCreateWindow(app->getSize().width, app->getSize().height, "RiftAR", nullptr, nullptr);
        if (!window)
            THROW_ERROR("Failed to create a window");
        glfwMakeContextCurrent(window);

        // Set up GL
        if (gl3wInit())
            THROW_ERROR("Failed to load GL3W");

        // Initialise the app
        app->init();

        // Main loop
        glfwSetKeyCallback(window, App::glfwKeyEvent);
        while (!glfwWindowShouldClose(window))
        {
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
//
// If the CONSOLE subsystem is chosen in the VS property pages, this function will just be ignored
// completely and main() will be used instead. The linker will link with the apps main generated by
// the DEFINE_MAIN macro.
#ifdef _WIN32
#include <cstdlib>              // __argc and __argv
int main(int, char**);
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
    return main(__argc, __argv);
}
#endif