#include "Common.h"
#include "lib/Model.h"
#include "camera/RealsenseCamera.h"
#include "KFusionTracker.h"

#include <TooN/se3.h>

#define KFUSION_DEBUG

#ifdef KFUSION_DEBUG
#include <kfusion/helpers.h>
#include <opencv2/highgui/highgui.hpp>

static Image<uchar4, HostDevice> lightModel;
const float3 light = make_float3(1, 1, -1.0);
const float3 ambient = make_float3(0.1, 0.1, 0.1);
#endif

// Helper functions
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

inline string toString(const glm::vec4& v)
{
    std::stringstream ss;
    ss << v.x << ", " << v.y << ", " << v.z << ", " << v.w;
    return ss.str();
}

KFusionTracker::KFusionTracker(RealsenseCamera* camera) :
    mSource(camera),
    mOptimiser(6, { 0.01, 0.01, 0.01, M_PI * 0.25, M_PI * 0.25, M_PI * 0.25 }),
    mNewOrigin(0.0f, 1.5f, 0.0f),
    mSearchTarget(nullptr)
{
    // Set up kfusion
    mKFusion = new KFusion;
    KFusionConfig config;
    float size = 1.5f;
    config.volumeSize = make_uint3(512);
    config.volumeDimensions = make_float3(size);
    config.nearPlane = 0.01f;
    config.farPlane = 5.0f;
    config.mu = 0.1f;
    config.maxweight = 100.0f;
    config.combinedTrackAndReduce = false;

    CameraIntrinsics& intr = camera->getIntrinsics(RealsenseCamera::DEPTH);
    config.inputSize = make_uint2(intr.width, intr.height);
    config.camera = make_float4(
        (float)intr.cameraMatrix.at<double>(0, 0),
        (float)intr.cameraMatrix.at<double>(1, 1),
        (float)intr.cameraMatrix.at<double>(0, 2),
        (float)intr.cameraMatrix.at<double>(1, 2));

    // config.iterations is a vector<int>, the length determines
    // the number of levels to be used in tracking
    // push back more then 3 iteration numbers to get more levels.
    config.iterations[0] = 10;
    config.iterations[1] = 5;
    config.iterations[2] = 4;

    mKFusion->Init(config);
    mKFusion->setPose(glmToKFusion(glm::translate(glm::mat4(), glm::vec3(size * 0.5f, size * 0.5f, 0.0f))));
    mDepthImage.alloc(make_uint2(intr.width, intr.height));

#ifdef KFUSION_DEBUG
    lightModel.alloc(make_uint2(intr.width, intr.height));
#endif

    mIsInitialising = true;
    mFrameCounter = 0;
}

KFusionTracker::~KFusionTracker()
{
}

void KFusionTracker::update(cv::Mat frame)
{
    // Give KFusion the depth data. It must be a uint16 in mm
    memcpy(mDepthImage.data(), frame.data, sizeof(uint16_t) * frame.total());
    cv::Mat depthImageWrapper(cv::Size2i(mDepthImage.size.x, mDepthImage.size.y), CV_16UC1, mDepthImage.data());
    depthImageWrapper *= mSource->getDepthScale() * 1000.0f;
    mKFusion->setKinectDeviceDepth(mDepthImage.getDeviceImage());

    // Integrate new data using KFusion module
    bool shouldIntegrate = mKFusion->Track();
    if (shouldIntegrate || mIsInitialising)
    {
        mKFusion->Integrate();
        mKFusion->Raycast();
        if (mFrameCounter > 2)
            mIsInitialising = false;
    }
    mFrameCounter++;
    CUDA_CHECK(cudaDeviceSynchronize());

    // Update the current pose
    mCameraPose = convKFusionCoordSystem(kfusionToGLM(mKFusion->pose));

    // Display integration state
    //cout << shouldIntegrate << " - " << toString(mCameraPose[3]) << endl;

    // Display KFusion state
#ifdef KFUSION_DEBUG
    renderLight(lightModel.getDeviceImage(), mKFusion->vertex, mKFusion->normal, light, ambient);
    CUDA_CHECK(cudaDeviceSynchronize());
    cv::imshow("KFusion Debug", cv::Mat(cv::Size2i(lightModel.size.x, lightModel.size.y), CV_8UC4, lightModel.data()));
#endif
}

void KFusionTracker::beginSearchingFor(Model* target)
{
    mSearchTarget = target;
}

bool KFusionTracker::checkTargetPosition(glm::mat4& resultTransform)
{
    if (!isSearching())
        return false;

    // Place the object in front of the camera
    glm::mat4 modelOffset = glm::translate(glm::mat4(), mSearchTarget->getSize() * glm::vec3(-0.5f, -0.5f, -2.5f)); // 2 x model depth away from the camera (assuming it's origin is centred)
    mSearchTarget->setTransform(mCameraPose * modelOffset);

    // Get the cost of the head model
    glm::mat4 flipMesh = glm::scale(glm::mat4(), glm::vec3(1.0f, -1.0f, -1.0f));
    glm::mat4 model = convKFusionCoordSystem(mSearchTarget->getTransform()) * flipMesh; // TODO: why is flipMesh required here?
    float cost = getCost(mSearchTarget, mKFusion->integration, model);
    cout << cost << endl;
    if (cost < 200.0f)
    {
        // Convert matrix into 6 parameters for the optimiser function
        std::vector<double> parameters;
        parameters.resize(6);
        CostFunction::mat4ToParameters(model, parameters);
        cout << parameters[0] << ", " << parameters[1] << ", " << parameters[2] << ", " << glm::degrees(parameters[3]) << ", " << glm::degrees(parameters[4]) << ", " << glm::degrees(parameters[5]) << endl;

        // The cost is low enough, lets optimise it further!
        mOptimiser.cost_function(std::shared_ptr<CostFunction>(new CostFunction(mSearchTarget, mKFusion->integration, this)));
        mOptimiser.init_parameters(parameters);
        drop::SimplexOptimizer::Termination term = mOptimiser.run();
        cout << "SimplexOptimizer terminated with code " << term << endl;

        // Optimisation done! Read the final transformation matrix
        parameters = mOptimiser.parameters();
        cout << parameters[0] << ", " << parameters[1] << ", " << parameters[2] << ", " << glm::degrees(parameters[3]) << ", " << glm::degrees(parameters[4]) << ", " << glm::degrees(parameters[5]) << endl;
        glm::mat4 location = CostFunction::mat4FromParameters(parameters);
        cout << "Original Cost: " << cost << endl << "Final cost: " << getCost(mSearchTarget, mKFusion->integration, location) << endl;

        // Set the result matrix
        resultTransform = convKFusionCoordSystem(location * flipMesh);
        mSearchTarget = nullptr;
        return true;
    }
    else
    {
        return false;
    }
}

bool KFusionTracker::isSearching() const
{
    return mSearchTarget != nullptr;
}

void KFusionTracker::reset()
{
    mKFusion->Reset();
    mIsInitialising = true;
    mFrameCounter = 0;
}

glm::mat4 KFusionTracker::getCameraPose() const
{
    return mCameraPose;
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
    if (scaledPos.x > 0 && scaledPos.y > 0 && scaledPos.z > 0 &&
        scaledPos.x < volume.size.x && scaledPos.y < volume.size.y && scaledPos.z < volume.size.z)
    {
        costs[index] = fmin(fabs(volume.interp(vertex)), trunc);
    }
    else
    {
        costs[index] = trunc;
    }
}

float KFusionTracker::getCost(Model* model, Volume volume, const glm::mat4& transform)
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

    // Multiply by 1000 here to convert from metres to millimetres, which helps with floating point
    // inaccuracy when using the drop simplex optimiser
    sum *= 1000.0f / count; 

    // Free memory and return
    delete[] vertices;
    delete[] costs;
    cudaFree(deviceVertices);
    cudaFree(deviceCosts);
    return sum;
}

glm::mat3 KFusionTracker::convKFusionCoordSystem(const glm::mat3& rotation) const
{
    glm::mat3 newRotation = rotation;
    newRotation[0][1] *= -1.0f;
    newRotation[0][2] *= -1.0f;
    newRotation[1][0] *= -1.0f;
    newRotation[2][0] *= -1.0f;
    return newRotation;
}

glm::mat4 KFusionTracker::convKFusionCoordSystem(const glm::mat4& transform) const
{
    glm::mat4 newOrientation = glm::mat4(convKFusionCoordSystem(glm::mat3(transform)));
    glm::mat4 newPosition = glm::translate(glm::mat4(), glm::vec3(transform[3]) * glm::vec3(1.0f, -1.0f, -1.0f) + mNewOrigin);
    return newPosition * newOrientation;
}

KFusionTracker::CostFunction::CostFunction(Model* model, Volume volume, KFusionTracker* tracker) :
    mModel(model),
    mVolume(volume),
    mTracker(tracker)
{
}

double KFusionTracker::CostFunction::evaluate(const std::vector<double> &parameters)
{
    return (double)mTracker->getCost(mModel, mVolume, mat4FromParameters(parameters));
}

glm::mat4 KFusionTracker::CostFunction::mat4FromParameters(const std::vector<double>& parameters)
{
    // Build matrix from the 6 parameters [x, y, z, rot-x, rot-y, rot-z]
    glm::mat4 translation = glm::translate(glm::mat4(), glm::vec3(parameters[0], parameters[1], parameters[2]));
    glm::mat4 rotation = glm::eulerAngleZ((float)parameters[5]) * glm::eulerAngleY((float)parameters[4]) * glm::eulerAngleX((float)parameters[3]);
    return translation * rotation;
}

bool floatEq(float a, float b)
{
    return fabs(a - b) < 1e-6f; // epsilon value
}

void KFusionTracker::CostFunction::mat4ToParameters(const glm::mat4& matrix, std::vector<double>& parameters)
{
    parameters[0] = matrix[3][0];
    parameters[1] = matrix[3][1];
    parameters[2] = matrix[3][2];

    glm::vec3 eulerAngles;
    const float PI = 3.1415927;
    if (floatEq(matrix[0][2], -1.0f))
    {
        eulerAngles.x = 0.0f;
        eulerAngles.y = PI * 0.5f;
        eulerAngles.z = atan2(matrix[1][0], matrix[2][0]);
    }
    else if (floatEq(matrix[0][2], 1.0f))
    {
        eulerAngles.x = 0.0f;
        eulerAngles.y = -PI * 0.5f;
        eulerAngles.z = atan2(-matrix[1][0], -matrix[2][0]);
    }
    else
    {
        float theta1 = -asin(matrix[0][2]);
        float theta2 = PI - theta1;

        float psi1 = atan2(matrix[1][2] / cos(theta1), matrix[2][2] / cos(theta1));
        float psi2 = atan2(matrix[1][2] / cos(theta2), matrix[2][2] / cos(theta2));

        float phi1 = atan2(matrix[0][1] / cos(theta1), matrix[0][0] / cos(theta1));
        float phi2 = atan2(matrix[0][1] / cos(theta2), matrix[0][0] / cos(theta2));

        if ((fabs(theta1) + fabs(psi1) + fabs(phi1)) < (fabs(theta2) + fabs(psi2) + fabs(phi2)))
        {
            eulerAngles.x = psi1;
            eulerAngles.y = theta1;
            eulerAngles.z = phi1;
        }
        else
        {
            eulerAngles.x = psi2;
            eulerAngles.y = theta2;
            eulerAngles.z = phi2;
        }
    }

    parameters[3] = eulerAngles.x;
    parameters[4] = eulerAngles.y;
    parameters[5] = eulerAngles.z;
}