#include "lib/Common.h"
#include "RiftAR.h"

#include "RiftOutput.h"
#include "DebugOutput.h"

#include <TooN/se3.h>

//#define RIFT_DISPLAY
//#define ENABLE_ZED

DEFINE_MAIN(RiftAR);

Image<uint16_t, HostDevice> depthImage;

Matrix4 glmToKFusion(const glm::mat4& mat)
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

glm::mat4 kfusionToGLM(const Matrix4& mat)
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

RiftAR::RiftAR()
{
}

void RiftAR::init()
{
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
    config.mu = 0.1;
    config.maxweight = 100.0f;
    config.combinedTrackAndReduce = false;

    CameraIntrinsics& intr = mRealsense->getIntrinsics(F200Camera::DEPTH);
    config.inputSize = make_uint2(intr.width, intr.height);
    config.camera = make_float4(intr.cameraMatrix.at<double>(0, 0), intr.cameraMatrix.at<double>(1, 1), intr.cameraMatrix.at<double>(0, 2), intr.cameraMatrix.at<double>(1, 2));

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
    mRenderCtx.projection = glm::perspective(glm::radians(75.0f), (float)destinationSize.width / (float)destinationSize.height, mRenderCtx.znear, mRenderCtx.zfar);
    mRenderCtx.model = new Model("../media/meshes/bob.stl");
    mRenderCtx.model->setPosition(glm::vec3(0.1f, -0.2f, -0.2f));
    //mRenderCtx.model->setScale(glm::vec3(2.0f));

    // Set up output
#ifdef ENABLE_ZED
    float fovH = mZed->getIntrinsics(ZEDCamera::LEFT).fovH;
    float fovV = mZed->getIntrinsics(ZEDCamera::LEFT).fovV;
    mRenderCtx.colourTextures[0] = mZed->getTexture(ZEDCamera::LEFT);
    mRenderCtx.colourTextures[1] = mZed->getTexture(ZEDCamera::RIGHT);
#else
    float fovH = mRealsense->getIntrinsics(F200Camera::COLOUR).fovH;
    float fovV = mRealsense->getIntrinsics(F200Camera::COLOUR).fovV;
    mRenderCtx.colourTextures[0] = mRealsense->getTexture(F200Camera::COLOUR);
    mRenderCtx.colourTextures[1] = mRealsense->getTexture(F200Camera::COLOUR);
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
    if (integrate || reset)
    {
        mKFusion->Integrate();
        mKFusion->Raycast();
        if (counter > 2)
            reset = false;
    }
    counter++;
    cudaDeviceSynchronize();

    // Copy over vertex information from the device
    static Image<float3, Host> vertices;
    if (vertices.size.x == 0)
        vertices.alloc(mKFusion->vertex.size);
    vertices = mKFusion->vertex;

    // DEBUG
    /*
    for (int y = 0; y < vertices.size.y; y++)
    {
        for (int x = 0; x < vertices.size.x; x++)
        {
            if ((x+y) % 50 != 0)
                continue;
            float3& vertex = vertices.data()[y * vertices.size.x + x];
            if (vertex.x != 0.0f)
                cout << x << ", " << y << " - " << vertex.x << ", " << vertex.y << ", " << vertex.z << endl;
        }
    }
    */

    // Display current position
    cout << integrate << " - " << glm::to_string(kfusionToGLM(mKFusion->pose)[3]) << endl;

    // Warp depth textures for occlusion
#ifdef ENABLE_ZED
    mRealsenseDepth->warpToPair(frame, mZedCalib,
        mRealsenseToZedLeft,
        mZed->getExtrinsics(ZEDCamera::LEFT, ZEDCamera::RIGHT) * mRealsenseToZedLeft);
#else
    mRealsenseDepth->warpToPair(frame, mZedCalib, mRealsenseToZedLeft, mRealsenseToZedLeft);
#endif

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
}
