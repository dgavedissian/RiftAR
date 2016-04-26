#include "Common.h"
#include "ZEDCamera.h"

ZEDCamera::ZEDCamera()
{
    mCamera = new sl::zed::Camera(sl::zed::ZEDResolution_mode::VGA, 100.0f);
    mWidth = mCamera->getImageSize().width;
    mHeight = mCamera->getImageSize().height;
    sl::zed::ERRCODE zederror = mCamera->init(sl::zed::MODE::PERFORMANCE, -1, true);
    if (zederror != sl::zed::SUCCESS)
        THROW_ERROR("ZED camera not detected");

    // Set up resources for each eye
    for (int eye = LEFT; eye <= RIGHT; eye++)
    {
        glGenTextures(1, &mTexture[eye]);
        glBindTexture(GL_TEXTURE_2D, mTexture[eye]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mWidth, mHeight, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr));
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

void ZEDCamera::copyFrameIntoCudaImage(Eye e, cudaGraphicsResource* resource)
{
    sl::zed::Mat m = mCamera->retrieveImage_gpu((sl::zed::SIDE)e);
    cudaArray_t arrIm;
    cudaGraphicsMapResources(1, &mCudaImage[e], 0);
    cudaGraphicsSubResourceGetMappedArray(&arrIm, mCudaImage[e], 0, 0);
    cudaMemcpy2DToArray(arrIm, 0, 0, m.data, m.step, mWidth * 4, mHeight, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &mCudaImage[e], 0);
}

void ZEDCamera::copyFrameIntoCVImage(Eye e, cv::Mat* mat)
{
    // Wrap the data in a cv::Mat then copy it. This const_cast is sadly necessary as the
    // getRawData interface isn't supposed to allow writes to this location of memory.
    // As we immediately copy the data, it shouldn't matter much here.
    cv::Mat wrapped(mWidth, mHeight, CV_8UC4, const_cast<void*>(getRawData(e)));
    cvtColor(wrapped, *mat, cv::COLOR_BGRA2RGB);
}

const void* ZEDCamera::getRawData(Eye e)
{
    return mCamera->retrieveImage((sl::zed::SIDE)e).data;
}

