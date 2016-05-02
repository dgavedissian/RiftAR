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
        mShowColour(false)
    {
    }

    void init() override
    {
        mZedCamera = new ZEDCamera();
        mRSCamera = new F200Camera(640, 480, 60, F200Camera::DEPTH);
        mRSCamera->setStream(F200Camera::DEPTH);

        // Create OpenGL images to view the depth stream
        glGenTextures(2, mDepth);
        glBindTexture(GL_TEXTURE_2D, mDepth[0]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, mRSCamera->getWidth(), mRSCamera->getHeight(), 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Load extrinsic parameters from the calibration
        cv::FileStorage fs("../stereo-params.xml", cv::FileStorage::READ);
        fs["R"] >> mR;
        fs["T"] >> mT;
        cout << "Extrinsics:" << endl;
        cout << "R: " << mR << endl;
        cout << "T: " << mT << endl;

        // Create objects
        mEyeQuad[0] = new Rectangle2D(glm::vec2(0.0f, 0.0f), glm::vec2(0.5f, 1.0f));
        mEyeQuad[1] = new Rectangle2D(glm::vec2(0.5f, 0.0f), glm::vec2(1.0f, 1.0f));
        mShader = new Shader("../media/quad.vs", "../media/quad.fs");
    }

    ~ViewCalibration()
    {
        delete mShader;
        delete mEyeQuad[0];
        delete mEyeQuad[1];

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
        mShader->bind();
        for (int i = 0; i < 2; i++)
        {
            mRSCamera->copyFrameIntoCVImage((CameraSource::Eye)i, &frame);

            // Create the output depth frame
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
            glm::mat3 rotateRsToZed = glm::inverse(convertCVToMat3<double>(mR));
            glm::vec3 transRsToZed = -convertCVToVec3<double>(mT);

            // Extrinsics that map from ZED left to ZED right
            float b = mZedCamera->getBaseline();
            float c = mZedCamera->getConvergence();
            glm::mat3 rotateLeftToRight(
                cos(c), 0.0f, -sin(c),
                0.0f, 1.0f, 0.0f,
                sin(c), 0.0f, cos(c)
            );
            glm::vec3 transLeftToRight(-b * cos(c * 0.5f), 0.0f, b * sin(c * 0.5f));

            // TODO: Clean this code up, and port to CUDA?
            for (int row = 0; row < frame.rows; row++)
            {
                for (int col = 0; col < frame.cols; col++)
                {
                    // Map point to 3D space
                    glm::vec3 point((float)col, (float)row, 1.0);
                    float depth = (float)frame.at<unsigned short>(row, col) * mRSCamera->getDepthScale();
                    point = (invRSCalib * point) * depth;

                    // Map from depth -> colour -> ZED left
                    point = rotateDepthToColour * point + transDepthToColour;
                    point = rotateRsToZed * point + transRsToZed;

                    // Map ZED left -> ZED right
                    if (i == 1)
                    {
                        point = rotateLeftToRight * point + transLeftToRight;
                    }

                    // Read the depth, and project the 3D point to the ZED camera
                    depth = point.z;
                    point = zedCalib * (point / depth);

                    // Write to the output texture
                    writeDepth(transformedDepth, point.x, point.y, depth);
                }
            }

            // Clean up the depth image
            //
            // Based on code from BackgroundMaskCleaner.cpp at
            // https://software.intel.com/en-us/blogs/2013/06/14/masking-rgb-inputs-with-depth-data-using-the-intel-percc-sdk-and-opencv
            static cv::Mat openElement = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10), cv::Point(5, 5));
            static cv::Mat closeElement = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6, 6), cv::Point(3, 3));
            morphologyEx(transformedDepth, transformedDepth, cv::MORPH_OPEN, openElement);
            morphologyEx(transformedDepth, transformedDepth, cv::MORPH_CLOSE, closeElement);

            // Display
            if (mShowColour)
            {
                glBindTexture(GL_TEXTURE_2D, mZedCamera->getTexture((CameraSource::Eye)i));
            }
            else
            {
                glBindTexture(GL_TEXTURE_2D, mDepth[0]);
                TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mRSCamera->getWidth(), mRSCamera->getHeight(), GL_RED, GL_UNSIGNED_SHORT, transformedDepth.ptr()));
            }
            mEyeQuad[i]->render();
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

    void writeDepth(cv::Mat& out, float x, float y, float depth)
    {
        cv::Point2i pixel(std::round(x), std::round(y));
        if (pixel.x >= 0 && pixel.x < out.cols && pixel.y >= 0 && pixel.y < out.rows)
        {
            unsigned short newDepth = (unsigned short)(depth / mRSCamera->getDepthScale());

            // Reject the pixel if we're trying to overwrite an object that is already in front of the existing pixel
            if (newDepth < out.at<unsigned short>(pixel.y, pixel.x))
            {
                out.at<unsigned short>(pixel.y, pixel.x) = newDepth;
            }
        }
    }

private:
    bool mShowColour;

    ZEDCamera* mZedCamera;
    F200Camera* mRSCamera;

    Rectangle2D* mEyeQuad[2];
    Shader* mShader;
    GLuint mDepth[2];

    // Extrinsics for the camera pair
    cv::Mat mR, mT;
};

DEFINE_MAIN(ViewCalibration);
