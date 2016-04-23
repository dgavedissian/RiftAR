#include "Common.h"
#include "ZEDCamera.h"

ZEDCamera::ZEDCamera()
{
    mCamera = new sl::zed::Camera(sl::zed::ZEDResolution_mode::VGA, 100.0f);
    mWidth = mCamera->getImageSize().width;
    mHeight = mCamera->getImageSize().height;
    sl::zed::ERRCODE zederror = mCamera->init(sl::zed::MODE::PERFORMANCE, -1, true);
    if (zederror != sl::zed::SUCCESS)
    {
        throw std::runtime_error("ZED camera not detected");
    }

    // Set up resources for each eye
    for (int eye = LEFT; eye <= RIGHT; eye++)
    {
        glGenTextures(1, &mTexture[eye]);
        glBindTexture(GL_TEXTURE_2D, mTexture[eye]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mWidth, mHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Set up CUDA graphics resource
        cudaError_t err = cudaGraphicsGLRegisterImage(&mCudaImage[eye], mTexture[eye], GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("ERROR: cannot create CUDA texture: " + std::to_string(err));
        }
    }
}

ZEDCamera::~ZEDCamera()
{
    delete mCamera;
}

void ZEDCamera::retrieve()
{
    if (mCamera->grab(sl::zed::SENSING_MODE::RAW, false, false))
    {
        //cout << "Error capturing frame from ZED" << endl;
        return;
        //throw std::runtime_error("Error capturing frame from ZED camera");
    }

    // Grab data
    for (int eye = LEFT; eye <= RIGHT; eye++)
    {
        sl::zed::Mat m = mCamera->retrieveImage_gpu((sl::zed::SIDE)eye);
        cudaArray_t arrIm;
        cudaGraphicsMapResources(1, &mCudaImage[eye], 0);
        cudaGraphicsSubResourceGetMappedArray(&arrIm, mCudaImage[eye], 0, 0);
        cudaMemcpy2DToArray(arrIm, 0, 0, m.data, m.step, mWidth * 4, mHeight, cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &mCudaImage[eye], 0);
    }
}