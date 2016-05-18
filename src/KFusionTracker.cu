#include "lib/Common.h"
#include "lib/Model.h"
#include "KFusionTracker.h"

const float3& reinterpretVec3AsFloat3(const glm::vec3& v)
{
    return *reinterpret_cast<const float3*>(&v);
}

__global__ void getCostForEachVertex(float* costs, float3* vertexData, Volume volume, Matrix4 transform)
{
    int index = threadIdx.x;

    // Transform vertex
    float3 vertex = transform * vertexData[index];

    // Calculate cell position in the volume and check bounds
    float trunc = 0.5f;
    int3 scaledPos = make_int3(
        vertex.x * volume.size.x / volume.dim.x,
        vertex.y * volume.size.y / volume.dim.y,
        vertex.z * volume.size.z / volume.dim.z);
    if (scaledPos.x >= 0 && scaledPos.y >= 0 && scaledPos.z >= 0 &&
        scaledPos.x < volume.size.x && scaledPos.y < volume.size.y && scaledPos.z < volume.size.z)
    {
        costs[index] = fmin(volume.interp(vertex), trunc);
    }
    else
    {
        costs[index] = trunc;
    }
}

#define COUNT 16

float getCost(Model* model, Volume volume, const glm::mat4& transform)
{
    // Select some vertices
    // TODO: Make this work for vertex count < 512
    assert(model->getVertices().size() >= COUNT);

    // Allocate space
    float3* vertices = new float3[COUNT];
    float* costs = new float[COUNT];
    float3* deviceVertices;
    float* deviceCosts;
    cudaMalloc(&deviceVertices, sizeof(float3) * COUNT);
    cudaMalloc(&deviceCosts, sizeof(float) * COUNT);

    // TODO: Randomly select these vertices
    for (int i = 0; i < COUNT; i++)
    {
        const glm::vec3& srcVertex = model->getVertices()[i];
        vertices[i].x = srcVertex.x;
        vertices[i].y = srcVertex.y;
        vertices[i].z = srcVertex.z;
    }

    // Copy to GPU memory
    CUDA_CHECK(cudaMemcpy(deviceVertices, vertices, sizeof(float3) * COUNT, cudaMemcpyHostToDevice));

    // Call kernel
    getCostForEachVertex<<<1, COUNT>>>(deviceCosts, deviceVertices, volume, glmToKFusion(transform));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy and sum results
    CUDA_CHECK(cudaMemcpy(costs, deviceCosts, sizeof(float) * COUNT, cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < COUNT; i++)
        sum += costs[i];

    // Free memory and return
    delete[] vertices;
    delete[] costs;
    cudaFree(deviceVertices);
    cudaFree(deviceCosts);
    return sum;
}
