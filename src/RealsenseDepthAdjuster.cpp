#include "lib/Common.h"
#include "RealsenseDepthAdjuster.h"

RealsenseDepthAdjuster::RealsenseDepthAdjuster(F200Camera* realsense, cv::Size destinationSize) :
    mRealsense(realsense),
    mColourSize(destinationSize)
{
    // Grab intrinsics for the realsense
    CameraIntrinsics& depthIntr = realsense->getIntrinsics(F200Camera::DEPTH);
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
}

void RealsenseDepthAdjuster::warpToPair(cv::Mat& frame, const glm::mat3& destCalib, const glm::mat4& leftExtrinsics, const glm::mat4& rightExtrinsics)
{
    static cv::Mat warpedFrame[2];

    // Initialise warped frames for each eye
    warpedFrame[0] = cv::Mat::zeros(cv::Size(mColourSize.width, mColourSize.height), CV_16UC1);
    warpedFrame[1] = cv::Mat::zeros(cv::Size(mColourSize.width, mColourSize.height), CV_16UC1);
    for (int c = 0; c < mColourSize.width; c++)
    {
        for (int r = 0; r < mColourSize.height; r++)
        {
            warpedFrame[0].at<uint16_t>(r, c) = 0xffff;
            warpedFrame[1].at<uint16_t>(r, c) = 0xffff;
        }
    }

    // Transform each pixel from the original frame using intrinsics and extrinsics
    glm::vec2 point;
    for (int row = 0; row < frame.rows; row++)
    {
        uint16_t* rowData = frame.ptr<uint16_t>(row);
        for (int col = 0; col < frame.cols; col++)
        {
            // Read depth
            uint16_t depthPixel = rowData[col];
            if (depthPixel == 0)
                continue;
            float depth = depthPixel * mRealsense->getDepthScale();
            float newDepth;

            // Warp to each eye
            for (int i = 0; i < 2; i++)
            {
                const glm::mat4& extrinsics = i == 0 ? leftExtrinsics : rightExtrinsics;

                // Top left of depth pixel
                point = glm::vec2((float)col - 0.5f, (float)row - 0.5f);
                newDepth = reprojectRealsenseToZed(point, depth, destCalib, extrinsics);
                cv::Point start((int)(point.x + 0.5f), (int)(point.y + 0.5f));

                // Bottom right of depth pixel
                point = glm::vec2((float)col + 0.5f, (float)row + 0.5f);
                newDepth = reprojectRealsenseToZed(point, depth, destCalib, extrinsics);
                cv::Point end((int)(point.x + 0.5f), (int)(point.y + 0.5f));

                // Swap start/end if appropriate
                if (start.x > end.x)
                    std::swap(start.x, end.x);
                if (start.y > end.y)
                    std::swap(start.y, end.y);

                // Reject pixels outside the target texture
                if (start.x < 0 || start.y < 0 || end.x >= warpedFrame[i].cols || end.y >= warpedFrame[i].rows)
                    continue;

                // Write the rectangle defined by the corners of the depth pixel to the output image
                for (int x = start.x; x <= end.x; x++)
                {
                    for (int y = start.y; y <= end.y; y++)
                        writeDepth(warpedFrame[i], x, y, newDepth);
                }
            }
        }
    }

    // Copy depth data
    for (int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, mDepthTextures[i]);
        GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mColourSize.width, mColourSize.height, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, warpedFrame[i].ptr()));
    }
}

GLuint RealsenseDepthAdjuster::getDepthTexture(uint eye)
{
    assert(eye < 2);
    return mDepthTextures[eye];
}

float RealsenseDepthAdjuster::reprojectRealsenseToZed(glm::vec2& point, float depth, const glm::mat3& destCalib, const glm::mat4& extrinsics)
{
    glm::vec3 homogenousPoint = glm::vec3(point, 1.0f);

    // De-project pixel to point in 3D space
    homogenousPoint.x = mRealsenseCalibInverse[0][0] * homogenousPoint.x + mRealsenseCalibInverse[2][0];
    homogenousPoint.y = mRealsenseCalibInverse[1][1] * homogenousPoint.y + mRealsenseCalibInverse[2][1];
    undistortRealsense(homogenousPoint, mRealsenseDistortCoeffs);
    homogenousPoint *= depth;

    // Map from Depth -> ZED
    homogenousPoint = glm::mat3(extrinsics) * homogenousPoint + glm::vec3(extrinsics[3]);

    // Project point to new pixel - conversion from vec4 to vec3 is equiv to multiplying by [I|0] matrix
    point.x = destCalib[0][0] * (homogenousPoint.x / homogenousPoint.z) + destCalib[2][0];
    point.y = destCalib[1][1] * (homogenousPoint.y / homogenousPoint.z) + destCalib[2][1];
    return homogenousPoint.z;
}

void RealsenseDepthAdjuster::writeDepth(cv::Mat& out, int x, int y, float depth)
{
    uint16_t oldDepth = out.at<uint16_t>(y, x);
    uint16_t newDepth = (uint16_t)(depth / mRealsense->getDepthScale());

    // Basic z-buffering here...
    if (newDepth < oldDepth || oldDepth == 0)
        out.at<uint16_t>(y, x) = newDepth;
}

void RealsenseDepthAdjuster::undistortRealsense(glm::vec3& point, const std::vector<double>& coeffs)
{
    float r2 = point.x * point.x + point.y * point.y;
    float f = 1.0f + coeffs[0] * r2 + coeffs[1] * r2 * r2 + coeffs[4] * r2 * r2 * r2;
    float ux = point.x * f + 2.0f * coeffs[2] * point.x * point.y + coeffs[3] * (r2 + 2.0f * point.x * point.x);
    float uy = point.y * f + 2.0f * coeffs[3] * point.x * point.y + coeffs[2] * (r2 + 2.0f * point.y * point.y);
    point.x = ux;
    point.y = uy;
}
