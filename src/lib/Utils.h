#pragma once

typedef unsigned int uint;

// Raise error
#define THROW_ERROR(m) throw std::runtime_error((std::string("Error in " __FILE__) + ":" + std::to_string(__LINE__) + "\n\n") + (m))

// Check GL
#define GL_CHECK(x) x; { GLenum error = glGetError(); if (error != GL_NO_ERROR) THROW_ERROR("OpenGL Error: " + std::to_string(error)); }

// Check CUDA
#define CUDA_CHECK(x) { cudaError_t error = x; if (error != cudaSuccess) THROW_ERROR("CUDA Error: " + string(cudaGetErrorString(error))); }

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
