#pragma once

#include <iostream>
#include <string>
#include <cstdint>
#include <memory>

using std::string;
using std::cout;
using std::cerr;
using std::endl;

using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::unique_ptr;

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

// OpenCV
#include <opencv2/opencv.hpp>

// Utils
#include "Utils.h"

// App
#include "App.h"
