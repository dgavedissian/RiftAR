#pragma once

#include <iostream>
#include <string>

using std::string;
using std::cout;
using std::cerr;
using std::endl;

// Windows Headers
#ifdef _WIN32
#define NOMINMAX				// Stop Windows.h from defining annoying min/max macros
#define WIN32_LEAN_AND_MEAN		// Strip out some unneded stuff from Windows.h
#include <Windows.h>			// WinMain function
#include <cstdlib>				// __argc and __argv
#endif

// GL and GLFW
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

// GLM
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat2x2.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
