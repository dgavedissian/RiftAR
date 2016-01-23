#ifdef _WIN32
#define NOMINMAX				// Stop Windows.h from defining annoying min/max macros
#define WIN32_LEAN_AND_MEAN		// Strip out some unneded stuff from Windows.h
#include <Windows.h>			// WinMain function
#include <cstdlib>				// __argc and __argv
#endif

// GLFW
#include <GLFW/glfw3.h>

// OVR
#include <OVR.h>

// RealSense Camera
#include "CameraRSF200.h"

// Application entry point
int main(int argc, char** argv)
{
	GLFWwindow* window;

	if (!glfwInit())
		return -1;

	// Set up the window
	window = glfwCreateWindow(640, 480, "Hello World", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Set up OVR
	ovrInitParams params;
	params.Flags = 0;
	params.RequestedMinorVersion = 0;
	params.LogCallback = nullptr;
	params.UserData = 0;
	params.ConnectionTimeoutMS = 0;
	ovr_Initialize(&params);
	ovrSession session;
	ovrGraphicsLuid luid;
	ovrResult result = ovr_Create(&session, &luid);
	if (OVR_FAILURE(result))
	{
		MessageBoxA(nullptr, "Oh noes! Something bad happened!", "Error", 0);
		return 1;
	}

	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	ovr_Destroy(session);
	ovr_Shutdown();
	glfwTerminate();
	return 0;
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