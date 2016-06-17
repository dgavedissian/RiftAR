#include "Common.h"
#include "RealsenseCamera.h"

#include <limits>

RealsenseCamera::RealsenseCamera(uint width, uint height, uint frameRate, uint streams) :
    mWidth(width),
    mHeight(height),
    mFrameRate(frameRate),
    mEnabledStreams(streams)
{
    initialiseDevice();

    // Initialise intrinsics and extrinsics
    for (int i = 0; i < STREAM_COUNT; i++)
    {
        if (isStreamEnabled(i))
        {
            mIntrinsics[i] = buildIntrinsics(i);
            for (int j = 0; j < STREAM_COUNT; j++)
            {
                if (isStreamEnabled(j))
                    mExtrinsics[i][j] = buildExtrinsics(i, j);
            }
        }
    }

    // Set up OpenGL
    if (streams & ENABLE_COLOUR)
    {
        // Initialise texture
        glGenTextures(1, &mStreamTextures[0]);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[0]);
        GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    if (streams & ENABLE_DEPTH)
    {
        glGenTextures(1, &mStreamTextures[1]);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[1]);
        GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_R16, width, height, 0, GL_RED, GL_UNSIGNED_SHORT, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    if (streams & ENABLE_INFRARED)
    {
        glGenTextures(1, &mStreamTextures[2]);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[2]);
        GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
}

RealsenseCamera::~RealsenseCamera()
{
    delete mContext;
}

void RealsenseCamera::capture()
{
    try
    {
        mDevice->wait_for_frames();
    }
    catch (rs::error&)
    {
        cout << "F200 was disconnected randomly due to tugging of the wire - attempting to reinitialise..." << endl;
        delete mContext;
        initialiseDevice();
    }
}

void RealsenseCamera::copyData()
{
    // Wrap the data in a cv::Mat then copy it. This const_cast is sadly necessary as the
    // getRawData interface isn't supposed to allow writes to this location of memory.
    // As we immediately copy the data, it shouldn't matter much here.
    if (mEnabledStreams & ENABLE_COLOUR)
    {
        CameraIntrinsics& intr = getIntrinsics(COLOUR);
        cv::Mat wrapped(intr.height, intr.width, CV_8UC3, const_cast<void*>(getRawData(COLOUR)));
        wrapped.copyTo(mStreamData[COLOUR]);
    }
    if (mEnabledStreams & ENABLE_DEPTH)
    {
        CameraIntrinsics& intr = getIntrinsics(DEPTH);
        cv::Mat wrapped(intr.height, intr.width, CV_16UC1, const_cast<void*>(getRawData(DEPTH)));
        wrapped.copyTo(mStreamData[DEPTH]);
    }
    if (mEnabledStreams & ENABLE_INFRARED)
    {
        CameraIntrinsics& intr = getIntrinsics(INFRARED);
        cv::Mat wrapped(intr.height, intr.width, CV_8UC1, const_cast<void*>(getRawData(INFRARED)));
        wrapped.copyTo(mStreamData[INFRARED]);
    }
}

void RealsenseCamera::updateTextures()
{
    if (mEnabledStreams & ENABLE_COLOUR)
    {
        CameraIntrinsics& intr = getIntrinsics(COLOUR);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[COLOUR]);
        GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, intr.width, intr.height, GL_BGR, GL_UNSIGNED_BYTE, mStreamData[COLOUR].ptr()));
    }
    if (mEnabledStreams & ENABLE_DEPTH)
    {
        CameraIntrinsics& intr = getIntrinsics(DEPTH);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[DEPTH]);
        GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, intr.width, intr.height, GL_RED, GL_UNSIGNED_SHORT, mStreamData[DEPTH].ptr()));
    }
    if (mEnabledStreams & ENABLE_INFRARED)
    {
        CameraIntrinsics& intr = getIntrinsics(INFRARED);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[INFRARED]);
        GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, intr.width, intr.height, GL_RED, GL_UNSIGNED_BYTE, mStreamData[INFRARED].ptr()));
    }
}

void RealsenseCamera::copyFrameIntoCVImage(uint camera, cv::Mat* mat)
{
    if (camera >= STREAM_COUNT)
        THROW_ERROR("Invalid stream");
    *mat = mStreamData[camera];
}

const void* RealsenseCamera::getRawData(uint camera)
{
    return mDevice->get_frame_data(mapCameraToStream(camera));
}

CameraIntrinsics RealsenseCamera::getIntrinsics(uint camera) const
{
    if (camera >= STREAM_COUNT)
        THROW_ERROR("Invalid stream");
    return mIntrinsics[camera];
}

glm::mat4 RealsenseCamera::getExtrinsics(uint camera1, uint camera2) const
{
    if (camera1 >= STREAM_COUNT || camera2 >= STREAM_COUNT)
        THROW_ERROR("Invalid stream");
    return mExtrinsics[camera1][camera2];
}

GLuint RealsenseCamera::getTexture(uint camera) const
{
    if (camera >= STREAM_COUNT)
        THROW_ERROR("Invalid stream");
    return mStreamTextures[camera];
}

bool RealsenseCamera::isStreamEnabled(uint camera) const
{
    if (camera >= STREAM_COUNT)
        THROW_ERROR("Invalid stream");
    return (mEnabledStreams & (1 << camera)) != 0;
}

void RealsenseCamera::initialiseDevice()
{
    rs::log_to_console(rs::log_severity::warn);
    mContext = new rs::context();
    if (mContext->get_device_count() == 0)
        THROW_ERROR("Unable to detect the RealsenseF200");
    mDevice = mContext->get_device(0);

    // Set up streams
    if (mEnabledStreams & ENABLE_COLOUR)
        mDevice->enable_stream(rs::stream::color, mWidth, mHeight, rs::format::bgr8, mFrameRate);
    if (mEnabledStreams & ENABLE_DEPTH)
        mDevice->enable_stream(rs::stream::depth, mWidth, mHeight, rs::format::z16, mFrameRate);
    if (mEnabledStreams & ENABLE_INFRARED)
        mDevice->enable_stream(rs::stream::infrared, mWidth, mHeight, rs::format::y8, mFrameRate);

    // Start the device
    mDevice->start();
}

rs::stream RealsenseCamera::mapCameraToStream(uint camera) const
{
    switch (camera)
    {
    case COLOUR: return rs::stream::color;
    case DEPTH: return rs::stream::depth;
    case INFRARED: return rs::stream::infrared;
    default: THROW_ERROR("Unknown camera ID");
    }
}

CameraIntrinsics RealsenseCamera::buildIntrinsics(uint camera) const
{
    rs::intrinsics& intr = mDevice->get_stream_intrinsics(mapCameraToStream(camera));
    CameraIntrinsics out;

    out.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    out.cameraMatrix.at<double>(0, 0) = intr.fx;
    out.cameraMatrix.at<double>(1, 1) = intr.fy;
    out.cameraMatrix.at<double>(0, 2) = intr.ppx;
    out.cameraMatrix.at<double>(1, 2) = intr.ppy;
    out.coeffs.insert(out.coeffs.end(), intr.coeffs, intr.coeffs + 5);

    float pi = (float)acos(-1.0);
    out.fovH = intr.hfov() * pi / 180.0f;
    out.fovV = intr.vfov() * pi / 180.0f;

    out.width = intr.width;
    out.height = intr.height;

    return out;
}

glm::mat4 RealsenseCamera::buildExtrinsics(uint camera1, uint camera2) const
{
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
