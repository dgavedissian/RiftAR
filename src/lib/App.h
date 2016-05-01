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
#define DEFINE_MAIN(C) int main(int argc, char** argv) { return runApp(argc, argv, new C); }
