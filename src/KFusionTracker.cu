#include "lib/Common.h"
#include "lib/Model.h"
#include "lib/F200Camera.h"
#include "KFusionTracker.h"

#include <TooN/se3.h>

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

KFusionTracker::KFusionTracker(F200Camera* camera) :
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

    CameraIntrinsics& intr = camera->getIntrinsics(F200Camera::DEPTH);
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
    static bool reset = true;
    static int counter = 0;
    if ((shouldIntegrate && isSearching()) || reset)
    {
        mKFusion->Integrate();
        mKFusion->Raycast();
        if (counter > 2)
            reset = false;
    }
    counter++;
    cudaDeviceSynchronize();

    // Update the current pose
    mCameraPose = convKFusionCoordSystem(kfusionToGLM(mKFusion->pose));

    // Display integration state
    //cout << shouldIntegrate << " - " << toString(mCameraPose[3]) << endl;
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
    glm::quat q = glm::quat_cast(rotation);

    // Mirror along x axis
    q.y *= -1.0f;
    q.z *= -1.0f;

    return glm::mat3_cast(q);
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
    // Build matrix from the 6 parameters [x, y, z, pitch, yaw, roll]
    glm::mat4 translation = glm::translate(glm::mat4(), glm::vec3(parameters[0], parameters[1], parameters[2]));
    glm::mat4 rotation = glm::yawPitchRoll((float)parameters[4], (float)parameters[3], (float)parameters[5]);
    return translation * rotation;
}

void KFusionTracker::CostFunction::mat4ToParameters(const glm::mat4& matrix, std::vector<double>& parameters)
{
    parameters[0] = matrix[3][0];
    parameters[1] = matrix[3][1];
    parameters[2] = matrix[3][2];

    glm::vec3 eulerAngles = glm::eulerAngles(glm::quat_cast(matrix));
    parameters[3] = eulerAngles.x;
    parameters[4] = eulerAngles.y;
    parameters[5] = eulerAngles.z;
}