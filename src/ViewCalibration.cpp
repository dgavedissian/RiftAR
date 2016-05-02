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

glm::vec3 convertCVToVec3(cv::Vec3f& v)
{
    return glm::make_vec3(v.val);
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
        mZedCamera = new ZEDCamera();
        mRSCamera = new F200Camera(640, 480, 60, F200Camera::DEPTH);
        mRSCamera->setStream(F200Camera::DEPTH);

        // Create OpenGL images to view the depth stream
        glGenTextures(1, &mDepth);
        glBindTexture(GL_TEXTURE_2D, mDepth);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, mRSCamera->getWidth(), mRSCamera->getHeight(), 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Load extrinsic parameters from the calibration
        cv::FileStorage fs("../stereo-params.xml", cv::FileStorage::READ);
        fs["R"] >> mR;
        fs["T"] >> mT;

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

        delete mZedCamera;
        delete mRSCamera;
    }

    void render() override
    {
        cv::Mat frame, transformedDepth;

        // Capture from cameras
        mZedCamera->capture();
        mZedCamera->updateTextures();
        mRSCamera->capture();

        // Display eyes
        for (int i = 0; i < 2; i++)
        {
            mRSCamera->copyFrameIntoCVImage((CameraSource::Eye)i, &frame);

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
            glm::mat3 invRSCalib = glm::inverse(convertCVToMat3<double>(mRSCamera->getCameraMatrix()));
            glm::mat3 zedCalib = convertCVToMat3<double>(mZedCamera->getCameraMatrix());

            // Extrinsics to map from depth to colour in the F200
            glm::mat3 rotateDepthToColour = convertCVToMat3<float>(mRSCamera->getRotationDepthToColour());
            glm::vec3 transDepthToColour = convertCVToVec3(mRSCamera->getTranslationDepthToColour());

            // Read extrinsics that map the ZED to the realsense colour camera, and invert to map in the opposite direction
            glm::mat3 rotateRsToZedLeft = glm::inverse(convertCVToMat3<double>(mR));
            glm::vec3 transRsToZedLeft = -convertCVToVec3<double>(mT);
            combineExtrinsics(rotateDepthToColour, rotateRsToZedLeft, transDepthToColour, transRsToZedLeft, rotateRsToZedLeft, transRsToZedLeft);

            // Combined extrinsics mapping RS depth to ZED
            glm::mat3 rotateRsToZed;
            glm::vec3 transRsToZed;
            if (i == 1) // right eye
            {
                // Extrinsics that map from ZED left to ZED right
                glm::vec3 transLeftToRight(-mZedCamera->getBaseline(), 0.0f, 0.0f);

                // Combine extrinsics
                combineExtrinsics(rotateRsToZedLeft, glm::mat3(1.0f), transRsToZedLeft, transLeftToRight, rotateRsToZed, transRsToZed);
            }
            else
            {
                rotateRsToZed = rotateRsToZedLeft;
                transRsToZed = transRsToZedLeft;
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
                    float depth = (float)depthPixel * mRSCamera->getDepthScale();

                    // Top left of depth pixel
                    glm::vec3 point((float)col - 0.5f, (float)row - 0.5f, 1.0);
                    // Deproject pixel to point
                    point = (invRSCalib * point) * depth;
                    // Map from Depth -> ZED
                    point = rotateRsToZed * point + transRsToZed;
                    // Project point
                    depth = point.z;
                    point = zedCalib * (point / depth);
                    cv::Point start(std::round(point.x), std::round(point.y));

                    // Bottom right of depth pixel
                    point.x = (float)col + 0.5f;
                    point.y = (float)row + 0.5f;
                    point.z = 1.0;
                    // Deproject pixel to point
                    point = (invRSCalib * point) * depth;
                    // Map from Depth -> ZED
                    point = rotateRsToZed * point + transRsToZed;
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

            // Display
            glViewport(i == 0 ? 0 : getSize().width / 2, 0, getSize().width / 2, getSize().height);
            if (!mShowColour)
            {
                // Show depth as a texture
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, mDepth);
                TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mRSCamera->getWidth(), mRSCamera->getHeight(), GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, transformedDepth.ptr()));
                mFullscreenShader->bind();
                mQuad->render();
            }
            else
            {
                // Show colour image using depth
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, mZedCamera->getTexture((CameraSource::Eye)i));
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, mDepth);
                TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mRSCamera->getWidth(), mRSCamera->getHeight(), GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, transformedDepth.ptr()));
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

    void combineExtrinsics(glm::mat3 r1, glm::mat3 r2, glm::vec3 t1, glm::vec3 t2, glm::mat3& ro, glm::vec3& to)
    {
        ro = r2 * r1;
        to = r2 * t1 + t2;
    }

    void writeDepth(cv::Mat& out, int x, int y, float depth)
    {
        unsigned short oldDepth = out.at<unsigned short>(y, x);
        unsigned short newDepth = (unsigned short)(depth / mRSCamera->getDepthScale());

        // Basic z-buffering here...
        if (newDepth < oldDepth)
        {
            out.at<unsigned short>(y, x) = newDepth;
        }
    }

private:
    bool mShowColour;

    ZEDCamera* mZedCamera;
    F200Camera* mRSCamera;

    Rectangle2D* mQuad;
    Shader* mFullscreenShader;
    Shader* mFullscreenWithDepthShader;
    GLuint mDepth;

    // Extrinsics for the camera pair
    cv::Mat mR, mT;
};

DEFINE_MAIN(ViewCalibration);
