#pragma once

class App
{
public:
    App() { msCurrentApp = this; }
    virtual ~App() {}

    virtual void init() = 0;
    virtual void render() = 0;
    virtual void keyEvent(int key, int scancode, int action, int mods) = 0;
    virtual cv::Size getSize() = 0;

    // C style callback which forwards the GLFW event to the application
    static App* msCurrentApp;
    static void glfwKeyEvent(GLFWwindow*, int key, int scancode, int action, int mods);
};

// An internal function which creates a GLFW window and executes the app
int runApp(int argc, char** argv, App* app);

// A handy macro used to define a main function which executes an implementation C of the App class
#define __DEFINE_MAIN(C) int main(int argc, char** argv) { return runApp(argc, argv, new C); }
#ifndef _WIN32
#   define DEFINE_MAIN(C) __DEFINE_MAIN(C)
#else
// Windows subsystem entry point
// If the CONSOLE subsystem is chosen in the VS property pages, this function will just be ignored
// completely and main() will be used instead.
#   define DEFINE_MAIN(C) __DEFINE_MAIN(C) int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) { return main(__argc, __argv); }
#endif
