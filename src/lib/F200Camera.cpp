#include "Common.h"
#include "F200Camera.h"

F200Camera::F200Camera(int width, int height, int frameRate, int stream)
{
    mWidth = width;
    mHeight = height;
    mContext = new rs::context();
    if (mContext->get_device_count() == 0)
        THROW_ERROR("Unable to detect the RealsenseF200");

    mDevice = mContext->get_device(0);
    if (stream & COLOUR || stream & DEPTH)
        mDevice->enable_stream(rs::stream::color, width, height, rs::format::rgba8, frameRate);
    if (stream & DEPTH)
        mDevice->enable_stream(rs::stream::depth, width, height, rs::format::z16, frameRate);
    if (stream & INFRARED)
        mDevice->enable_stream(rs::stream::infrared, width, height, rs::format::y8, frameRate);
    if (stream & INFRARED2)
        mDevice->enable_stream(rs::stream::infrared2, width, height, rs::format::y8, frameRate);
    mDevice->start();

    // Get intrinsic parameters
    if (stream & COLOUR || stream & DEPTH)
    {
        rs::intrinsics& intr = mDevice->get_stream_intrinsics(rs::stream::depth);
        mCameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        mCameraMatrix.at<double>(0, 0) = intr.fx;
        mCameraMatrix.at<double>(1, 1) = intr.fy;
        mCameraMatrix.at<double>(0, 2) = intr.ppx;
        mCameraMatrix.at<double>(1, 2) = intr.ppx;
        mDistCoeffs.insert(mDistCoeffs.end(), intr.coeffs, intr.coeffs + 5);
    }

    // Get extrinsics from depth to colour
    rs::extrinsics& extr = mDevice->get_extrinsics(rs::stream::depth, rs::stream::color);
    mRotateDToC = cv::Mat(3, 3, CV_32F);
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            mRotateDToC.at<float>(row, col) = extr.rotation[col * 3 + row];
        }
    }
    mTranslateDToC(0) = extr.translation[0];
    mTranslateDToC(1) = extr.translation[1];
    mTranslateDToC(2) = extr.translation[2];

    // Set up image
    if (stream & COLOUR)
    {
        glGenTextures(1, &mStreamTextures[0]);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[0]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    if (stream & DEPTH)
    {
        glGenTextures(1, &mStreamTextures[1]);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[1]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_UNSIGNED_SHORT, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    if (stream & INFRARED)
    {
        glGenTextures(1, &mStreamTextures[2]);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[2]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    if (stream & INFRARED2)
    {
        glGenTextures(1, &mStreamTextures[3]);
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[3]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
}

F200Camera::~F200Camera()
{
    delete mContext;
}

void F200Camera::setStream(Stream stream)
{
    mCurrentStream = stream;

    // Update the texture IDs to allow the views to be rendered in OpenGL
    switch (mCurrentStream)
    {
    case COLOUR:
        mTexture[0] = mTexture[1] = mStreamTextures[0];
        break;

    case DEPTH:
        mTexture[0] = mTexture[1] = mStreamTextures[1];
        break;

    case INFRARED:
        mTexture[0] = mTexture[1] = mStreamTextures[2];
        break;

    case INFRARED2:
        mTexture[0] = mTexture[1] = mStreamTextures[3];
        break;

    default:
        break;
    }

}

void F200Camera::capture()
{
    mDevice->wait_for_frames();
}

void F200Camera::updateTextures()
{
    // Grab data
    // TODO: Use a pixel buffer object instead of glTexSubImage2D?
    switch (mCurrentStream)
    {
    case COLOUR:
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[0]);
        TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, getRawData(LEFT)));
        break;

    case DEPTH:
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[1]);
        TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RED, GL_UNSIGNED_SHORT, getRawData(LEFT)));
        break;

    case INFRARED:
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[2]);
        TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RED, GL_UNSIGNED_SHORT, getRawData(LEFT)));
        break;

    case INFRARED2:
        glBindTexture(GL_TEXTURE_2D, mStreamTextures[3]);
        TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RED, GL_UNSIGNED_SHORT, getRawData(LEFT)));
        break;

    default:
        break;
    }
}

void F200Camera::copyFrameIntoCudaImage(Eye e, cudaGraphicsResource* resource)
{
    THROW_ERROR("Unimplemented");
}

void F200Camera::copyFrameIntoCVImage(Eye e, cv::Mat* mat)
{
    // Wrap the data in a cv::Mat then copy it. This const_cast is sadly necessary as the
    // getRawData interface isn't supposed to allow writes to this location of memory.
    // As we immediately copy the data, it shouldn't matter much here.
    if (mCurrentStream == COLOUR)
    {
        cv::Mat wrapped(mHeight, mWidth, CV_8UC4, const_cast<void*>(getRawData(e)));
        cvtColor(wrapped, *mat, cv::COLOR_RGBA2BGR);
    }
    else if (mCurrentStream == DEPTH)
    {
        cv::Mat wrapped(mHeight, mWidth, CV_16UC1, const_cast<void*>(getRawData(e)));
        wrapped.copyTo(*mat);
    }
    else
    {
        cv::Mat wrapped(mHeight, mWidth, CV_8UC1, const_cast<void*>(getRawData(e)));
        wrapped.copyTo(*mat);
    }
}

const void* F200Camera::getRawData(Eye e)
{
    switch (mCurrentStream)
    {
    case COLOUR:
        return mDevice->get_frame_data(rs::stream::color);

    case DEPTH:
        return mDevice->get_frame_data(rs::stream::depth);

    case INFRARED:
        return mDevice->get_frame_data(rs::stream::infrared);

    case INFRARED2:
        return mDevice->get_frame_data(rs::stream::infrared2);

    default:
        return nullptr;
    }
}
