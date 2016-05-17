#pragma once

#include <iostream>
#include <string>
#include <cstdint>

using std::string;
using std::cout;
using std::cerr;
using std::endl;

typedef unsigned int uint;

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

#ifndef __CUDACC__
#include <glm/gtx/string_cast.hpp>
#endif

// OpenCV
#include <opencv2/opencv.hpp>

// Raise error
#define THROW_ERROR(m) throw std::runtime_error((std::string("Error in " __FILE__) + ":" + std::to_string(__LINE__) + "\n\n") + (m))

// Check GL
#define GL_CHECK(x) x; { GLenum error = glGetError(); if (error != GL_NO_ERROR) THROW_ERROR("OpenGL Error: " + std::to_string(error)); }

// Check CUDA
#define CUDA_CHECK(x) { cudaError_t error = x; if (error != cudaSuccess) THROW_ERROR("CUDA Error: " + string(cudaGetErrorString(error))); }

// App
#include "App.h"

// Utils
template <class T>
glm::mat3 convertCVToMat3(cv::Mat& m)
{
    // GLM is column major but OpenCV is row major
    glm::mat3 out;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            out[j][i] = (float)m.at<T>(i, j);
        }
    }
    return out;
}

template <class T>
glm::vec3 convertCVToVec3(cv::Mat& v)
{
    return glm::vec3((float)v.at<T>(0), (float)v.at<T>(1), (float)v.at<T>(2));
}
