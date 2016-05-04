#include "lib/Common.h"
#include "lib/F200Camera.h"
#include "lib/ZEDCamera.h"
#include "lib/Rectangle2D.h"
#include "lib/Shader.h"

#include <opencv2/imgproc/imgproc.hpp>

template <class T>
glm::mat3 convertCVToMat3(cv::Mat& m)
{
    // GLM is column major but OpenCV is row major
    glm::mat3 out;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            out[j][i] = (float)m.at<T>(i, j);
        }
    }
    return out;
}

template <class T>
glm::vec3 convertCVToVec3(cv::Mat& v)
{
    return glm::vec3((float)v.at<T>(0), (float)v.at<T>(1), (float)v.at<T>(2));
}

class ViewCalibration : public App
{
public:
    ViewCalibration() :
        mShowColour(true)
    {
    }

    void init() override
    {
        mZed = new ZEDCamera();
        mRealsense = new F200Camera(640, 480, 60, F200Camera::ENABLE_COLOUR | F200Camera::ENABLE_DEPTH);

        // Create OpenGL images to view the depth stream
        glGenTextures(1, &mDepth);
        glBindTexture(GL_TEXTURE_2D, mDepth);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
            mRealsense->getWidth(F200Camera::COLOUR), mRealsense->getHeight(F200Camera::COLOUR),
            0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Read extrinsics parameters that map the ZED to the realsense colour camera, and invert to map in the opposite direction
        cv::FileStorage fs("../stereo-params.xml", cv::FileStorage::READ);
        cv::Mat R, T;
        fs["R"] >> R;
        fs["T"] >> T;
        mRSColourToZedLeft.rotation = glm::inverse(convertCVToMat3<double>(R));
        mRSColourToZedLeft.translation = -convertCVToVec3<double>(T);

        // Create objects
        mQuad = new Rectangle2D(glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 1.0f));
        mFullscreenShader = new Shader("../media/quad.vs", "../media/quad_inv.fs");
        mFullscreenWithDepthShader = new Shader("../media/quad.vs", "../media/quad_inv_depth.fs");
        mFullscreenWithDepthShader->bind();
        mFullscreenWithDepthShader->setUniform<int>("rgbCameraImage", 0);
        mFullscreenWithDepthShader->setUniform<int>("depthCameraImage", 1);
    }

    ~ViewCalibration()
    {
        delete mFullscreenWithDepthShader;
        delete mFullscreenShader;
        delete mQuad;

        delete mZed;
        delete mRealsense;
    }

    void render() override
    {
        cv::Mat frame, transformedDepth;

        // Capture from cameras
        mZed->capture();
        mZed->updateTextures();
        mRealsense->capture();

        // Display eyes
        for (int i = 0; i < 2; i++)
        {
            mRealsense->copyFrameIntoCVImage(F200Camera::DEPTH, &frame);

            // Create the output depth frame and initialise to maximum depth. This is required for the morphology filters
            transformedDepth = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_16UC1);
            for (int c = 0; c < frame.cols; c++)
            {
                for (int r = 0; r < frame.rows; r++)
                {
                    transformedDepth.at<unsigned short>(r, c) = 0xffff;
                }
            }

            // Read parameters
            glm::mat3 rsCalib = convertCVToMat3<double>(mRealsense->getIntrinsics(F200Camera::DEPTH).cameraMatrix);
            glm::mat3 invRSCalib = glm::inverse(rsCalib);
            glm::mat3 zedCalib = convertCVToMat3<double>(mRealsense->getIntrinsics(ZEDCamera::LEFT).cameraMatrix);

            // Extrinsics to map from depth to colour in the F200
            CameraExtrinsics depthToColour = mRealsense->getExtrinsics(F200Camera::DEPTH, F200Camera::COLOUR);

            // Combined extrinsics mapping RS depth to ZED
            CameraExtrinsics rsToZed = CameraExtrinsics::combine(depthToColour, mRSColourToZedLeft);
            if (i == 1) // Right eye
            {
                rsToZed = CameraExtrinsics::combine(rsToZed, mZed->getExtrinsics(ZEDCamera::LEFT, ZEDCamera::RIGHT));
            }

            // TODO: Clean this code up, and port to CUDA?
            for (int row = 0; row < frame.rows; row++)
            {
                for (int col = 0; col < frame.cols; col++)
                {
                    // Read depth
                    unsigned short depthPixel = frame.at<unsigned short>(row, col);
                    if (depthPixel == 0)
                        continue;
                    float depth = (float)depthPixel * mRealsense->getDepthScale();

                    // Top left of depth pixel
                    glm::vec3 point((float)col - 0.5f, (float)row - 0.5f, 1.0);
                    // Deproject pixel to point
                    point = (invRSCalib * point) * depth;
                    // Map from Depth -> ZED
                    point = rsToZed.rotation * point + rsToZed.translation;
                    // Project point
                    depth = point.z;
                    point = zedCalib * (point / depth);
                    cv::Point start(std::round(point.x), std::round(point.y));

                    // Bottom right of depth pixel
                    point = glm::vec3((float)col + 0.5, (float)row + 0.5f, 1.0);
                    // Deproject pixel to point
                    point = (invRSCalib * point) * depth;
                    // Map from Depth -> ZED
                    point = rsToZed.rotation * point + rsToZed.translation;
                    // Project point
                    depth = point.z;
                    point = zedCalib * (point / depth);
                    cv::Point end(std::round(point.x), std::round(point.y));

                    // Swap start/end if appropriate
                    if (start.x > end.x)
                        std::swap(start.x, end.x);
                    if (start.y > end.y)
                        std::swap(start.y, end.y);

                    // Reject pixels outside the target texture
                    if (start.x < 0 || start.y < 0 || end.x >= frame.cols || end.y >= frame.rows)
                        continue;

                    // Write the rectangle defined by the corners of the depth pixel to the output image
                    for (int x = start.x; x <= end.x; x++)
                    {
                        for (int y = start.y; y <= end.y; y++)
                        {
                            writeDepth(transformedDepth, x, y, depth);
                        }
                    }
                }
            }

            // Copy depth data
            glBindTexture(GL_TEXTURE_2D, mDepth);
            TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                mRealsense->getWidth(F200Camera::COLOUR), mRealsense->getHeight(F200Camera::COLOUR),
                GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, transformedDepth.ptr()));

            // Display
            glViewport(i == 0 ? 0 : getSize().width / 2, 0, getSize().width / 2, getSize().height);
            if (!mShowColour)
            {
                // Show depth as a texture
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, mDepth);
                mFullscreenShader->bind();
                mQuad->render();
            }
            else
            {
                // Show colour image using depth
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, mZed->getTexture(i));
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, mDepth);
                mFullscreenWithDepthShader->bind();
                mQuad->render();
            }
        }
    }

    void keyEvent(int key, int scancode, int action, int mods) override
    {
        if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
            mShowColour = !mShowColour;
    }

    cv::Size getSize() override
    {
        return cv::Size(1280, 480);
    }

    void writeDepth(cv::Mat& out, int x, int y, float depth)
    {
        unsigned short oldDepth = out.at<unsigned short>(y, x);
        unsigned short newDepth = (unsigned short)(depth / mRealsense->getDepthScale());

        // Basic z-buffering here...
        if (newDepth < oldDepth)
        {
            out.at<unsigned short>(y, x) = newDepth;
        }
    }

private:
    bool mShowColour;

    ZEDCamera* mZed;
    F200Camera* mRealsense;

    Rectangle2D* mQuad;
    Shader* mFullscreenShader;
    Shader* mFullscreenWithDepthShader;
    GLuint mDepth;

    // Extrinsics for the camera pair
    CameraExtrinsics mRSColourToZedLeft;
};

DEFINE_MAIN(ViewCalibration);
