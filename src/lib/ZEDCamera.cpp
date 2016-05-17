#include "Common.h"
#include "ZEDCamera.h"

ZEDCamera::ZEDCamera(sl::zed::ZEDResolution_mode resolution, int fps)
{
    mCamera = new sl::zed::Camera(resolution, (float)fps);
    sl::zed::ERRCODE zederror = mCamera->init(sl::zed::MODE::NONE, -1, true);
    if (zederror != sl::zed::SUCCESS)
        THROW_ERROR("ZED camera not detected");

    // Get intrinsics for left camera
    // From testing, the intrinsics for the left and right cameras were identical
    sl::zed::StereoParameters* params = mCamera->getParameters();
    sl::zed::CamParameters& left = params->LeftCam;
    mIntrinsics.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    mIntrinsics.cameraMatrix.at<double>(0, 0) = left.fx;
    mIntrinsics.cameraMatrix.at<double>(1, 1) = left.fy;
    mIntrinsics.cameraMatrix.at<double>(0, 2) = left.cx;
    mIntrinsics.cameraMatrix.at<double>(1, 2) = left.cy;
    mIntrinsics.coeffs.insert(mIntrinsics.coeffs.end(), left.disto, left.disto + 5);

    // Image size
    mIntrinsics.width = mCamera->getImageSize().width;
    mIntrinsics.height = mCamera->getImageSize().height;

    // Fov
    mIntrinsics.fovH = atanf(mCamera->getImageSize().width / (mCamera->getParameters()->LeftCam.fx * 2.0f)) * 2.0f;
    mIntrinsics.fovV = atanf(mCamera->getImageSize().height / (mCamera->getParameters()->LeftCam.fy * 2.0f)) * 2.0f;

    // Set up resources for each eye
    for (int eye = LEFT; eye <= RIGHT; eye++)
    {
        glGenTextures(1, &mTexture[eye]);
        glBindTexture(GL_TEXTURE_2D, mTexture[eye]);
        GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mIntrinsics.width, mIntrinsics.height, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Set up CUDA graphics resource
        cudaError_t err = cudaGraphicsGLRegisterImage(&mCudaImage[eye], mTexture[eye], GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
        if (err != cudaSuccess)
            THROW_ERROR("ERROR: cannot create CUDA texture: " + std::to_string(err));
    }
}

ZEDCamera::~ZEDCamera()
{
    delete mCamera;
}

void ZEDCamera::capture()
{
    if (mCamera->grab(sl::zed::SENSING_MODE::RAW, false, false))
    {
        //cout << "Error capturing frame from ZED" << endl;
    }
}

void ZEDCamera::updateTextures()
{
    copyFrameIntoCudaImage(LEFT, mCudaImage[LEFT]);
    copyFrameIntoCudaImage(RIGHT, mCudaImage[RIGHT]);
}

void ZEDCamera::copyFrameIntoCudaImage(uint camera, cudaGraphicsResource* resource)
{
    sl::zed::Mat m = mCamera->retrieveImage_gpu(mapCameraToSide(camera));
    cudaArray_t arrIm;
    cudaGraphicsMapResources(1, &mCudaImage[camera], 0);
    cudaGraphicsSubResourceGetMappedArray(&arrIm, mCudaImage[camera], 0, 0);
    cudaMemcpy2DToArray(arrIm, 0, 0, m.data, m.step, mIntrinsics.width * 4, mIntrinsics.height, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &mCudaImage[camera], 0);
}

void ZEDCamera::copyFrameIntoCVImage(uint camera, cv::Mat* mat)
{
    cv::Mat wrapped = slMat2cvMat(mCamera->retrieveImage(mapCameraToSide(camera)));
    cvtColor(wrapped, *mat, cv::COLOR_BGRA2BGR);
}

const void* ZEDCamera::getRawData(uint camera)
{
    return mCamera->retrieveImage(mapCameraToSide(camera)).data;
}

CameraIntrinsics ZEDCamera::getIntrinsics(uint camera) const
{
    return mIntrinsics;
}

glm::mat4 ZEDCamera::getExtrinsics(uint camera1, uint camera2) const
{
    if (camera1 > 1 || camera2 > 1)
        THROW_ERROR("Camera must be ZEDCamera::LEFT or ZEDCamera::Right");

    // Mapping a camera to itself
    if (camera1 == camera2)
        return glm::mat4();

    // Baseline maps right to left
    float baseline = mCamera->getParameters()->baseline * 1e-3f;

    // Left to right
    if (camera2 == ZEDCamera::RIGHT)
        baseline = -baseline;

    return glm::translate(glm::mat4(), glm::vec3(baseline, 0.0f, 0.0f));
}

GLuint ZEDCamera::getTexture(uint camera) const
{
    if (camera > 1)
        THROW_ERROR("Camera must be ZEDCamera::LEFT or ZEDCamera::Right");
    return mTexture[camera];
}

sl::zed::SIDE ZEDCamera::mapCameraToSide(uint camera) const
{
    if (camera > 1)
        THROW_ERROR("Unsupported camera");
    return (sl::zed::SIDE)camera;
}

