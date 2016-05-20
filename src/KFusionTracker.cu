#include "lib/Common.h"
#include "lib/Model.h"
#include "KFusionTracker.h"

const float3& reinterpretVec3AsFloat3(const glm::vec3& v)
{
    return *reinterpret_cast<const float3*>(&v);
}

__global__ void getCostForEachVertex(float* costs, float3* vertexData, int vertexCount, Volume volume, Matrix4 transform)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= vertexCount)
        return;

    // Transform vertex
    float3 vertex = transform * vertexData[index];

    // Calculate cell position in the volume and check bounds
    float trunc = 0.25f;
    int3 scaledPos = make_int3(
        vertex.x * volume.size.x / volume.dim.x,
        vertex.y * volume.size.y / volume.dim.y,
        vertex.z * volume.size.z / volume.dim.z);
    if (scaledPos.x >= 0 && scaledPos.y >= 0 && scaledPos.z >= 0 &&
        scaledPos.x < volume.size.x && scaledPos.y < volume.size.y && scaledPos.z < volume.size.z)
    {
        costs[index] = fmin(fabs(volume.interp(vertex)), trunc);
    }
    else
    {
        costs[index] = trunc;
    }
}

float getCost(Model* model, Volume volume, const glm::mat4& transform)
{
    // Select some vertices
    int stride = 10;
    int count = model->getVertices().size() / stride;

    // Allocate space
    float3* vertices = new float3[count];
    float* costs = new float[count];
    float3* deviceVertices;
    float* deviceCosts;
    cudaMalloc(&deviceVertices, sizeof(float3) * count);
    cudaMalloc(&deviceCosts, sizeof(float) * count);

    // Select vertices
    for (int i = 0; i < count; i++)
    {
        const glm::vec3& srcVertex = model->getVertices()[i * stride];
        vertices[i].x = srcVertex.x;
        vertices[i].y = srcVertex.y;
        vertices[i].z = srcVertex.z;
    }

    // Copy to GPU memory
    CUDA_CHECK(cudaMemcpy(deviceVertices, vertices, sizeof(float3) * count, cudaMemcpyHostToDevice));

    // Call kernel
    dim3 blockSize(1024);
    dim3 blockCount((count + blockSize.x - 1) / blockSize.x);
    getCostForEachVertex<<<blockCount, blockSize>>>(deviceCosts, deviceVertices, count, volume, glmToKFusion(transform));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy and sum results
    CUDA_CHECK(cudaMemcpy(costs, deviceCosts, sizeof(float) * count, cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < count; i++)
        sum += costs[i];
    sum *= 1000.0f / count;

    // Free memory and return
    delete[] vertices;
    delete[] costs;
    cudaFree(deviceVertices);
    cudaFree(deviceCosts);
    return sum;
}
