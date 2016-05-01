#pragma once

#include <iostream>
#include <string>

using std::string;
using std::cout;
using std::cerr;
using std::endl;

// Windows Headers
#ifdef _WIN32
#define NOMINMAX                // Stop Windows.h from defining annoying min/max macros
#define WIN32_LEAN_AND_MEAN     // Strip out some unneded stuff from Windows.h
#include <Windows.h>            // WinMain function
#include <cstdlib>              // __argc and __argv
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

// OpenCV
#include <opencv2/opencv.hpp>

// Raise error
#define THROW_ERROR(m) throw std::runtime_error((std::string("Error in " __FILE__) + ":" + std::to_string(__LINE__) + "\n\n") + (m))

// Check GL
#define TEST_GL(x) x; { GLenum error = glGetError(); if (error != GL_NO_ERROR) THROW_ERROR("OpenGL Error: " + std::to_string(error)); }

// App
#include "App.h"
