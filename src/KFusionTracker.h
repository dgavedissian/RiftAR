#pragma once

#include <kfusion/kfusion.h>

class Model;

inline Matrix4 glmToKFusion(const glm::mat4& mat)
{
    // KFusion's Matrix4 is row major, whilst glm is column major
    Matrix4 out;
    for (int i = 0; i < 4; i++)
    {
        out.data[i].x = mat[0][i];
        out.data[i].y = mat[1][i];
        out.data[i].z = mat[2][i];
        out.data[i].w = mat[3][i];
    }
    return out;
}

inline glm::mat4 kfusionToGLM(const Matrix4& mat)
{
    // KFusion's Matrix4 is row major, whilst glm is column major
    glm::mat4 out;
    for (int i = 0; i < 4; i++)
    {
        out[0][i] = mat.data[i].x;
        out[1][i] = mat.data[i].y;
        out[2][i] = mat.data[i].z;
        out[3][i] = mat.data[i].w;
    }
    return out;
}

// Sum of the distance values of each vertex looked up in KFusions signed distance field
float getCost(Model* model, Volume volume, const glm::mat4& transform);
