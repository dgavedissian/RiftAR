#include "Common.h"

App* App::msCurrentApp = nullptr;

void App::keyEvent(int key, int scancode, int action, int mods)
{
}

void App::mouseButtonEvent(int button, int action, int mods)
{
}

void App::scrollEvent(double x, double y)
{
}

void App::glfwKeyEvent(GLFWwindow*, int key, int scancode, int action, int mods)
{
    msCurrentApp->keyEvent(key, scancode, action, mods);
}

void App::glfwMouseButtonEvent(GLFWwindow*, int button, int action, int mods)
{
    msCurrentApp->mouseButtonEvent(button, action, mods);
}

void App::glfwScrollEvent(GLFWwindow*, double x, double y)
{
    msCurrentApp->scrollEvent(x, y);
}
