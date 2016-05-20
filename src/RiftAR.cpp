#include "lib/Common.h"
#include "RiftAR.h"

#include "RiftOutput.h"
#include "DebugOutput.h"

#include "KFusionTracker.h"

#include <TooN/se3.h>

#include "simplex/dropSimplex.h"

//#define RIFT_DISPLAY
//#define ENABLE_ZED

DEFINE_MAIN(RiftAR);

Image<uint16_t, HostDevice> depthImage;
drop::SimplexOptimizer optimiser(6, { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 });

class KFusionCostFunction : public drop::CostFunctionSimplex
{
public:
    KFusionCostFunction(Model* model, Volume volume) :
        mModel(model),
        mVolume(volume)
    {
    }

    virtual double evaluate(const std::vector<double> &parameters)
    {
        return (double)getCost(mModel, mVolume, mat4FromParameters(parameters));
    }

    static glm::mat4 mat4FromParameters(const std::vector<double>& parameters)
    {
        // Build matrix from the 6 parameters [x, y, z, pitch, yaw, roll]
        glm::mat4 translation = glm::translate(glm::mat4(), glm::vec3(parameters[0], parameters[1], parameters[2]));
        glm::mat4 rotation = glm::yawPitchRoll((float)parameters[4], (float)parameters[3], (float)parameters[5]);
        return translation * rotation;
    }

    static void mat4ToParameters(const glm::mat4& matrix, std::vector<double>& parameters)
    {
        parameters[0] = matrix[3][0];
        parameters[1] = matrix[3][1];
        parameters[2] = matrix[3][2];

        glm::vec3 eulerAngles = glm::eulerAngles(glm::quat_cast(matrix));
        parameters[3] = eulerAngles.x;
        parameters[4] = eulerAngles.y;
        parameters[5] = eulerAngles.z;
    }

private:
    Model* mModel;
    Volume mVolume;

};

glm::vec3 newOrigin(0.0f, 1.5f, 0.0f);

glm::mat3 convKFusionCoordSystem(const glm::mat3& rotation)
{
    glm::quat q = glm::quat_cast(rotation);

    // Mirror along x axis
    q.y *= -1.0f;
    q.z *= -1.0f;

    return glm::mat3_cast(q);
}

glm::mat4 convKFusionCoordSystem(const glm::mat4& transform)
{
    glm::mat4 newOrientation = glm::mat4(convKFusionCoordSystem(glm::mat3(transform)));
    glm::mat4 newPosition = glm::translate(glm::mat4(), glm::vec3(transform[3]) * glm::vec3(1.0f, -1.0f, -1.0f) + newOrigin);
    return newPosition * newOrientation;
}

RiftAR::RiftAR()
{
}

void RiftAR::init()
{
    mRenderCtx.model = nullptr;

    // Set up the cameras
#ifdef ENABLE_ZED
    mZed = new ZEDCamera(sl::zed::HD720, 60);
#endif
    mRealsense = new F200Camera(640, 480, 60, F200Camera::ENABLE_COLOUR | F200Camera::ENABLE_DEPTH);

    // Get the width/height of the output colour stream that the user sees
    cv::Size destinationSize;
#ifdef ENABLE_ZED
    destinationSize.width = mZed->getWidth(ZEDCamera::LEFT);
    destinationSize.height = mZed->getHeight(ZEDCamera::LEFT);
#else
    destinationSize.width = mRealsense->getWidth(F200Camera::COLOUR);
    destinationSize.height = mRealsense->getHeight(F200Camera::COLOUR);
#endif

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

    CameraIntrinsics& intr = mRealsense->getIntrinsics(F200Camera::DEPTH);
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
    depthImage.alloc(make_uint2(intr.width, intr.height));

    // Set up depth warping
    setupDepthWarpStream(destinationSize);

    // Set up scene
    mRenderCtx.backbufferSize = getSize();
    mRenderCtx.depthScale = USHRT_MAX * mRealsense->getDepthScale();
    mRenderCtx.znear = 0.01f;
    mRenderCtx.zfar = 10.0f;
    mRenderCtx.model = new Model("../media/meshes/bob.stl");

    // Set up output
#ifdef ENABLE_ZED
    float fovH = mZed->getIntrinsics(ZEDCamera::LEFT).fovH;
    float fovV = mZed->getIntrinsics(ZEDCamera::LEFT).fovV;
    mRenderCtx.colourTextures[0] = mZed->getTexture(ZEDCamera::LEFT);
    mRenderCtx.colourTextures[1] = mZed->getTexture(ZEDCamera::RIGHT);
    mRenderCtx.projection = mZed->getIntrinsics(ZEDCamera::LEFT).buildGLProjection(mRenderCtx.znear, mRenderCtx.zfar);
#else
    float fovH = mRealsense->getIntrinsics(F200Camera::COLOUR).fovH;
    float fovV = mRealsense->getIntrinsics(F200Camera::COLOUR).fovV;
    mRenderCtx.colourTextures[0] = mRealsense->getTexture(F200Camera::COLOUR);
    mRenderCtx.colourTextures[1] = mRealsense->getTexture(F200Camera::COLOUR);
    mRenderCtx.projection = mRealsense->getIntrinsics(F200Camera::COLOUR).buildGLProjection(mRenderCtx.znear, mRenderCtx.zfar);
#endif

#ifdef ENABLE_ZED
    bool invertColours = true;
#else
    bool invertColours = false;
#endif
#ifdef RIFT_DISPLAY
    mOutputCtx = new RiftOutput(getSize(), fovH, fovV, invertColours);
#else
    mOutputCtx = new DebugOutput(mRenderCtx, invertColours);
#endif

    // Enable culling and depth testing
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
}

RiftAR::~RiftAR()
{
    if (mRenderCtx.model)
        delete mRenderCtx.model;

#ifdef ENABLE_ZED
    delete mZed;
#endif
    delete mRealsense;

    delete mOutputCtx;
}

void RiftAR::render()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Update the textures
#ifdef ENABLE_ZED
    mZed->capture();
    mZed->updateTextures();
#endif
    mRealsense->capture();
    mRealsense->updateTextures();

    // Read depth texture data
    static cv::Mat frame;
    mRealsense->copyFrameIntoCVImage(F200Camera::DEPTH, &frame);

    // Give KFusion the depth data. It must be a uint16 in mm
    memcpy(depthImage.data(), frame.data, sizeof(uint16_t) * frame.total());
    cv::Mat depthImageWrapper(cv::Size2i(depthImage.size.x, depthImage.size.y), CV_16UC1, depthImage.data());
    depthImageWrapper *= mRealsense->getDepthScale() * 1000.0f;
    mKFusion->setKinectDeviceDepth(depthImage.getDeviceImage());

    // Integrate new data using KFusion module
    bool integrate = mKFusion->Track();
    static bool reset = true;
    static int counter = 0;
    static int successfulIntegrations = 0;
    if ((integrate && successfulIntegrations < 100) || reset)
    {
        mKFusion->Integrate();
        mKFusion->Raycast();
        if (counter > 2)
            reset = false;
        successfulIntegrations++;
    }
    counter++;
    cudaDeviceSynchronize();

    // Display current position
    glm::mat4 cameraPose = convKFusionCoordSystem(kfusionToGLM(mKFusion->pose));
    mRenderCtx.view = glm::inverse(cameraPose);

    // Update the position of the head model
    static bool foundTransform = false;
    if (!foundTransform)
    {
        glm::mat4 headOffset = glm::translate(glm::mat4(), glm::vec3(-mRenderCtx.model->getSize().x * 0.5f, -mRenderCtx.model->getSize().y * 0.5f, -0.5f));
        mRenderCtx.model->setTransform(cameraPose * headOffset);

        // Get the cost of the head model
        glm::mat4 flipMesh = glm::scale(glm::mat4(), glm::vec3(1.0f, -1.0f, -1.0f));
        glm::mat4 model = convKFusionCoordSystem(mRenderCtx.model->getTransform()) * flipMesh; // TODO: why is flipMesh required here?
        float cost = getCost(mRenderCtx.model, mKFusion->integration, model);
        if (cost < 0.15f)
        {
            // Convert matrix into 6 parameters for the optimiser function
            std::vector<double> parameters;
            parameters.resize(6);
            KFusionCostFunction::mat4ToParameters(model, parameters);
            cout << parameters[0] << ", " << parameters[1] << ", " << parameters[2] << ", " << glm::degrees(parameters[3]) << ", " << glm::degrees(parameters[4]) << ", " << glm::degrees(parameters[5]) << endl;

            // The cost is low enough, lets optimise it further!
            optimiser.cost_function(std::shared_ptr<KFusionCostFunction>(new KFusionCostFunction(mRenderCtx.model, mKFusion->integration)));
            optimiser.init_parameters(parameters);
            optimiser.run();

            // Optimisation done! What parameters did we get?
            parameters = optimiser.parameters();
            cout << parameters[0] << ", " << parameters[1] << ", " << parameters[2] << ", " << glm::degrees(parameters[3]) << ", " << glm::degrees(parameters[4]) << ", " << glm::degrees(parameters[5]) << endl;
            glm::mat4 location = KFusionCostFunction::mat4FromParameters(parameters);

            foundTransform = true;
            mRenderCtx.model->setTransform(convKFusionCoordSystem(location * flipMesh));
        }
    }

    // Warp depth textures for occlusion
    mRealsenseDepth->warpToPair(frame, mZedCalib, mRenderCtx.eyeMatrix[0], mRenderCtx.eyeMatrix[1]);

    // Render scene
    mOutputCtx->renderScene(mRenderCtx);
}

void RiftAR::keyEvent(int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_SPACE)
        {
            DebugOutput* debug = dynamic_cast<DebugOutput*>(mOutputCtx);
            if (debug)
                debug->toggleDebug();
        }

        if (key == GLFW_KEY_R)
        {
            mKFusion->Reset();
        }
    }
}

cv::Size RiftAR::getSize()
{
    return cv::Size(1600, 600);
}

// Distortion
void RiftAR::setupDepthWarpStream(cv::Size destinationSize)
{
    mRealsenseDepth = new RealsenseDepthAdjuster(mRealsense, destinationSize);
    mRenderCtx.depthTextures[0] = mRealsenseDepth->getDepthTexture(0);
    mRenderCtx.depthTextures[1] = mRealsenseDepth->getDepthTexture(1);

    // Read parameters
#ifdef ENABLE_ZED
    mZedCalib = convertCVToMat3<double>(mZed->getIntrinsics(ZEDCamera::LEFT).cameraMatrix);
#else
    mZedCalib = convertCVToMat3<double>(mRealsense->getIntrinsics(F200Camera::COLOUR).cameraMatrix);
#endif

    // Read extrinsics parameters that map the ZED to the realsense colour camera, and invert
    // to map in the opposite direction
#ifdef ENABLE_ZED
    cv::FileStorage fs("../stereo-params.xml", cv::FileStorage::READ);
    cv::Mat rotationMatrix, translation;
    fs["R"] >> rotationMatrix;
    fs["T"] >> translation;
    glm::mat4 realsenseColourToZedLeft = buildExtrinsic(
        glm::inverse(convertCVToMat3<double>(rotationMatrix)),
        -convertCVToVec3<double>(translation));
#else
    glm::mat4 realsenseColourToZedLeft;
#endif

    // Extrinsics to map from depth to colour in the F200
    glm::mat4 depthToColour = mRealsense->getExtrinsics(F200Camera::DEPTH, F200Camera::COLOUR);

    // Combined extrinsics mapping realsense depth to ZED left
    mRealsenseToZedLeft = realsenseColourToZedLeft * depthToColour;

#ifdef ENABLE_ZED
    mRenderCtx.eyeMatrix[0] = mRealsenseToZedLeft;
    mRenderCtx.eyeMatrix[1] = mZed->getExtrinsics(ZEDCamera::LEFT, ZEDCamera::RIGHT) * mRealsenseToZedLeft;
#else
    mRenderCtx.eyeMatrix[0] = mRenderCtx.eyeMatrix[1] = mRealsenseToZedLeft;
#endif
}
