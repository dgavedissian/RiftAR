#pragma once

#include <iostream>
#include <string>
#include <cstdint>

using std::string;
using std::cout;
using std::cerr;
using std::endl;

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
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>

#ifndef __CUDACC__
#include <glm/gtx/string_cast.hpp>
#endif

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// OpenCV
#include <opencv2/opencv.hpp>

// Utils taken from the framework
#include "../lib/Utils.h"
