#include "Common.h"
#include "F200Camera.h"

#include <limits>

#define USE_DEPTH_ALIGNED_TO_COLOUR

F200Camera::F200Camera(uint width, uint height, uint frameRate, uint streams) :
    mEnabledStreams(streams)
{
    rs::log_to_console(rs::log_severity::warn);
    mContext = new rs::context();
    if (mContext->get_device_count() == 0)
        THROW_ERROR("Unable to detect the RealsenseF200");
    mDevice = mContext->get_device(0);
    
    // Set up streams
    if (streams & ENABLE_COLOUR)
    {
        mDevice->enable_stream(rs::stream::color, width, height, rs::format::rgba8, frameRate);
        glGenTextures(1, &mStreamTextures[0]);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[0]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    if (streams & ENABLE_DEPTH)
    {
        mDevice->enable_stream(rs::stream::depth, width, height, rs::format::z16, frameRate);
        glGenTextures(1, &mStreamTextures[1]);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[1]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_UNSIGNED_SHORT, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    if (streams & ENABLE_INFRARED)
    {
        mDevice->enable_stream(rs::stream::infrared, width, height, rs::format::y8, frameRate);
        glGenTextures(1, &mStreamTextures[2]);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[2]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    if (streams & ENABLE_INFRARED2)
    {
        mDevice->enable_stream(rs::stream::infrared2, width, height, rs::format::y8, frameRate);
        glGenTextures(1, &mStreamTextures[3]);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[3]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    // Start the device
    mDevice->start();
}

F200Camera::~F200Camera()
{
    delete mContext;
}

void F200Camera::capture()
{
    try
    {
        mDevice->wait_for_frames();
    }
    catch (std::exception&)
    {
        // Ignore
    }
}

void F200Camera::updateTextures()
{
    // Grab data
    // TODO: Use a pixel buffer object instead of glTexSubImage2D?
    if (mEnabledStreams & ENABLE_COLOUR)
    {
        CameraIntrinsics& intr = getIntrinsics(COLOUR);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[COLOUR]);
        TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, intr.width, intr.height, GL_RGBA, GL_UNSIGNED_BYTE, getRawData(COLOUR)));
    }
    if (mEnabledStreams & ENABLE_DEPTH)
    {
        CameraIntrinsics& intr = getIntrinsics(DEPTH);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[DEPTH]);
        TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, intr.width, intr.height, GL_RED, GL_UNSIGNED_SHORT, getRawData(DEPTH)));
    }
    if (mEnabledStreams & ENABLE_INFRARED)
    {
        CameraIntrinsics& intr = getIntrinsics(INFRARED);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[INFRARED]);
        TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, intr.width, intr.height, GL_RED, GL_UNSIGNED_BYTE, getRawData(INFRARED)));
    }
    if (mEnabledStreams & ENABLE_INFRARED2)
    {
        CameraIntrinsics& intr = getIntrinsics(INFRARED2);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[INFRARED2]);
        TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, intr.width, intr.height, GL_RED, GL_UNSIGNED_BYTE, getRawData(INFRARED2)));
    }
}

void F200Camera::copyFrameIntoCudaImage(uint camera, cudaGraphicsResource* resource)
{
    THROW_ERROR("Unimplemented");
}

void F200Camera::copyFrameIntoCVImage(uint camera, cv::Mat* mat)
{
    // Wrap the data in a cv::Mat then copy it. This const_cast is sadly necessary as the
    // getRawData interface isn't supposed to allow writes to this location of memory.
    // As we immediately copy the data, it shouldn't matter much here.
    CameraIntrinsics& intr = getIntrinsics(camera);
    if (camera == COLOUR)
    {
        cv::Mat wrapped(intr.height, intr.width, CV_8UC4, const_cast<void*>(getRawData(camera)));
        cvtColor(wrapped, *mat, cv::COLOR_RGBA2BGR);
    }
    else if (camera == DEPTH)
    {
        cv::Mat wrapped(intr.height, intr.width, CV_16UC1, const_cast<void*>(getRawData(camera)));
        wrapped.copyTo(*mat);
    }
    else
    {
        cv::Mat wrapped(intr.height, intr.width, CV_8UC1, const_cast<void*>(getRawData(camera)));
        wrapped.copyTo(*mat);
    }
}

const void* F200Camera::getRawData(uint camera)
{
#ifdef USE_DEPTH_ALIGNED_TO_COLOUR
    if (camera == DEPTH)
        return mDevice->get_frame_data(rs::stream::depth_aligned_to_color);
#endif
    return mDevice->get_frame_data(mapCameraToStream(camera));
}

CameraIntrinsics F200Camera::getIntrinsics(uint camera) const
{
#ifdef USE_DEPTH_ALIGNED_TO_COLOUR
    if (camera == DEPTH)
        return buildIntrinsics(mDevice->get_stream_intrinsics(rs::stream::color));
#endif
    return buildIntrinsics(mDevice->get_stream_intrinsics(mapCameraToStream(camera)));
}

glm::mat4 F200Camera::getExtrinsics(uint camera1, uint camera2) const
{
#ifdef USE_DEPTH_ALIGNED_TO_COLOUR
    if (camera1 == DEPTH)
        camera1 = COLOUR;
    if (camera2 == DEPTH)
        camera2 = COLOUR;
#endif

    // Mapping a camera to itself
    if (camera1 == camera2)
        return glm::mat4();

    glm::mat4 out;
    rs::extrinsics& extr = mDevice->get_extrinsics(mapCameraToStream(camera1), mapCameraToStream(camera2));
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            out[col][row] = extr.rotation[col * 3 + row];
        }
    }
    out[3] = glm::vec4(extr.translation[0], extr.translation[1], extr.translation[2], 1.0f);
    return out;
}

GLuint F200Camera::getTexture(uint camera) const
{
    return mStreamTextures[camera];
}

rs::stream F200Camera::mapCameraToStream(uint camera) const
{
    switch (camera)
    {
    case COLOUR: return rs::stream::color;
    case DEPTH: return rs::stream::depth;
    case INFRARED: return rs::stream::infrared;
    case INFRARED2: return rs::stream::infrared2;
    default: THROW_ERROR("Unknown camera ID");
    }
}

CameraIntrinsics F200Camera::buildIntrinsics(rs::intrinsics& intr) const
{
    CameraIntrinsics out;

    out.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    out.cameraMatrix.at<double>(0, 0) = intr.fx;
    out.cameraMatrix.at<double>(1, 1) = intr.fy;
    out.cameraMatrix.at<double>(0, 2) = intr.ppx;
    out.cameraMatrix.at<double>(1, 2) = intr.ppx; // Why must this be x?
    out.coeffs.insert(out.coeffs.end(), intr.coeffs, intr.coeffs + 5);

    float pi = (float)acos(-1.0);
    out.fovH = intr.hfov() * pi / 180.0f;
    out.fovV = intr.vfov() * pi / 180.0f;

    out.width = intr.width;
    out.height = intr.height;

    return out;
}
