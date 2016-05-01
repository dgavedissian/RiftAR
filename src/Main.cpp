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