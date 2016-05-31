#include "Common.h"
#include "RealsenseDepthAdjuster.h"

struct DistortCoeffs
{
    float c[5];
};

__device__ uint2 generatePos2()
{
    return make_uint2(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y);
}

__device__ void undistortRealsense(glm::vec3& point, const DistortCoeffs& coeffs)
{
    float r2 = point.x * point.x + point.y * point.y;
    float f = 1.0f + coeffs.c[0] * r2 + coeffs.c[1] * r2 * r2 + coeffs.c[4] * r2 * r2 * r2;
    float ux = point.x * f + 2.0f * coeffs.c[2] * point.x * point.y + coeffs.c[3] * (r2 + 2.0f * point.x * point.x);
    float uy = point.y * f + 2.0f * coeffs.c[3] * point.x * point.y + coeffs.c[2] * (r2 + 2.0f * point.y * point.y);
    point.x = ux;
    point.y = uy;
}

__device__ float reprojectRealsenseToZed(glm::vec2& point, float depth, const glm::mat3& srcCalib, const DistortCoeffs& srcCoeffs, const glm::mat3& destCalib, const glm::mat4& extrinsics)
{
    glm::vec3 homogenousPoint = glm::vec3(point, 1.0f);

    // De-project pixel to point in 3D space
    homogenousPoint.x = srcCalib[0][0] * homogenousPoint.x + srcCalib[2][0];
    homogenousPoint.y = srcCalib[1][1] * homogenousPoint.y + srcCalib[2][1];
    undistortRealsense(homogenousPoint, srcCoeffs);
    homogenousPoint *= depth;

    // Map from Depth -> ZED
    homogenousPoint = glm::mat3(extrinsics) * homogenousPoint + glm::vec3(extrinsics[3]);

    // Project point to new pixel - conversion from vec4 to vec3 is equiv to multiplying by [I|0] matrix
    point.x = destCalib[0][0] * (homogenousPoint.x / homogenousPoint.z) + destCalib[2][0];
    point.y = destCalib[1][1] * (homogenousPoint.y / homogenousPoint.z) + destCalib[2][1];
    return homogenousPoint.z;
}

template <class T>
__device__ void swap(T& a, T& b)
{
    T temp = b;
    b = a;
    a = temp;
}

__global__ void warpToCamera(DeviceImage<uint16_t> src, DeviceImage<uint32_t> dest, float depthScale, glm::mat3 srcCalib, DistortCoeffs srcCoeffs, glm::mat3 destCalib, glm::mat4 extrinsics)
{
    glm::vec2 point;
    float newDepth;

    // Generate position based on block and thread
    uint2 index = generatePos2();
    if (index.x >= src.size.x || index.y >= src.size.y)
        return;

    // Read depth and skip invalid depth values
    uint16_t depthPixel = src[index];
    if (depthPixel == 0)
        return;
    float depth = depthPixel * depthScale;

    // Top left of depth pixel
    point = glm::vec2((float)index.x - 0.5f, (float)index.y - 0.5f);
    newDepth = reprojectRealsenseToZed(point, depth, srcCalib, srcCoeffs, destCalib, extrinsics);
    int2 start = make_int2((int)(point.x + 0.5f), (int)(point.y + 0.5f));

    // Bottom right of depth pixel
    point = glm::vec2((float)index.x + 0.5f, (float)index.y + 0.5f);
    newDepth = reprojectRealsenseToZed(point, depth, srcCalib, srcCoeffs, destCalib, extrinsics);
    int2 end = make_int2((int)(point.x + 0.5f), (int)(point.y + 0.5f));

    // Swap start/end if appropriate
    if (start.x > end.x)
        swap(start.x, end.x);
    if (start.y > end.y)
        swap(start.y, end.y);

    // Reject pixels outside the target texture
    if (start.x < 0 || start.y < 0 || end.x >= dest.size.x || end.y >= dest.size.y)
        return;

    // Write the rectangle defined by the corners of the depth pixel to the output image
    for (int x = start.x; x <= end.x; x++)
    {
        for (int y = start.y; y <= end.y; y++)
        {
            uint16_t toWrite = (uint16_t)(newDepth / depthScale);
            atomicMin(&dest[make_uint2(x, y)], (uint32_t)toWrite);
        }
    }
}

__global__ void initDepthMaps(DeviceImage<uint32_t> depth)
{
    uint2 index = generatePos2();
    if (index.x >= depth.size.x || index.y >= depth.size.y)
        return;
    depth[index] = 0xffff;
}

__global__ void flattenIntToShort(DeviceImage<uint32_t> src, DeviceImage<uint16_t> dest)
{
    uint2 index = generatePos2();
    if (index.x >= src.size.x || index.y >= src.size.y)
        return;
    dest[index] = (uint16_t)src[index];
}

RealsenseDepthAdjuster::RealsenseDepthAdjuster(RealsenseCamera* realsense, cv::Size destinationSize) :
    mRealsense(realsense),
    mColourSize(destinationSize)
{
    // Grab intrinsics for the realsense
    CameraIntrinsics depthIntr = realsense->getIntrinsics(RealsenseCamera::DEPTH);
    mRealsenseCalibInverse = glm::inverse(convertCVToMat3<double>(depthIntr.cameraMatrix));
    mRealsenseDistortCoeffs = depthIntr.coeffs;

    // Create OpenGL images to view the depth stream
    glGenTextures(2, mDepthTextures);
    for (int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, mDepthTextures[i]);
        GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
            mColourSize.width, mColourSize.height,
            0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
}

RealsenseDepthAdjuster::~RealsenseDepthAdjuster()
{
    freeCuda();
}

void RealsenseDepthAdjuster::warpToPair(cv::Mat& frame, const glm::mat3& destCalib, const glm::mat4& leftExtrinsics, const glm::mat4& rightExtrinsics)
{
    // Allocate memory
    if (frame.cols != mSrcImage.size.x || frame.rows != mSrcImage.size.y)
    {
        freeCuda();
        allocCuda(cv::Size(frame.cols, frame.rows));
    }

    // Copy frame to GPU memory
    CUDA_CHECK(cudaMemcpy(mSrcImage.data, frame.ptr<uint16_t>(), frame.cols * frame.rows * sizeof(uint16_t), cudaMemcpyHostToDevice));

    // Set up kernel block sizes
    dim3 blockDim(32, 32);
    dim3 srcGridDim((frame.cols + blockDim.x - 1) / blockDim.x, (frame.rows + blockDim.y - 1) / blockDim.y);
    dim3 destGridDim((mColourSize.width + blockDim.x - 1) / blockDim.x, (mColourSize.height + blockDim.y - 1) / blockDim.y);

    // Initialise temp images to 0xffff
    initDepthMaps<<<destGridDim, blockDim>>>(mTempImage[0]);
    initDepthMaps<<<destGridDim, blockDim>>>(mTempImage[1]);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Set up distortion coefficients
    DistortCoeffs coeffs;
    for (int i = 0; i < 5; i++)
        coeffs.c[i] = (float)mRealsenseDistortCoeffs[i];

    // Warp to temporary storage
    warpToCamera<<<srcGridDim, blockDim>>>(mSrcImage, mTempImage[0], mRealsense->getDepthScale(), mRealsenseCalibInverse, coeffs, destCalib, leftExtrinsics);
    warpToCamera<<<srcGridDim, blockDim>>>(mSrcImage, mTempImage[1], mRealsense->getDepthScale(), mRealsenseCalibInverse, coeffs, destCalib, rightExtrinsics);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cast uint32 to uint16
    flattenIntToShort<<<destGridDim, blockDim>>>(mTempImage[0], mDestImage[0]);
    flattenIntToShort<<<destGridDim, blockDim>>>(mTempImage[1], mDestImage[1]);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy depth data
    for (int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, mDepthTextures[i]);
        GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mColourSize.width, mColourSize.height, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, mDestImageHost[i]));
    }
}

GLuint RealsenseDepthAdjuster::getDepthTexture(uint eye)
{
    assert(eye < 2);
    return mDepthTextures[eye];
}

void RealsenseDepthAdjuster::allocCuda(cv::Size srcSize)
{
    // Source image data
    mSrcImage.size.x = srcSize.width;
    mSrcImage.size.y = srcSize.height;
    cudaMalloc(&mSrcImage.data, mSrcImage.size.x * mSrcImage.size.y * sizeof(uint16_t));

    // Intermediate image data
    for (int i = 0; i < 2; i++)
    {
        mTempImage[i].size.x = mColourSize.width;
        mTempImage[i].size.y = mColourSize.height;
        cudaMalloc(&mTempImage[i].data, mColourSize.width * mColourSize.height * sizeof(uint32_t));
    }

    // Destination image data
    for (int i = 0; i < 2; i++)
    {
        mDestImage[i].size.x = mColourSize.width;
        mDestImage[i].size.y = mColourSize.height;
        cudaHostAlloc(&mDestImageHost[i], mColourSize.width * mColourSize.height * sizeof(uint16_t), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&mDestImage[i].data, mDestImageHost[i], 0);
    }
}

void RealsenseDepthAdjuster::freeCuda()
{
    cudaFree(mSrcImage.data);
    cudaFree(mTempImage[0].data);
    cudaFree(mTempImage[1].data);
    cudaFree(mDestImageHost[0]);
    cudaFree(mDestImageHost[1]);
}
