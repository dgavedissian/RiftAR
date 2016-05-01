#include "lib/Common.h"
#include "lib/F200Camera.h"
#include "lib/ZEDCamera.h"
#include "lib/Rectangle2D.h"
#include "lib/Shader.h"

template <class T>
glm::mat3 convertCVToMat3(cv::Mat& m)
{
    glm::mat3 out;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            out[j][i] = m.at<T>(i, j);
        }
    }
    return out;
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

        // Create OpenGL images to view the colour stream
        glGenTextures(2, mColour);
        glBindTexture(GL_TEXTURE_2D, mColour[0]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mZedCamera->getWidth(), mZedCamera->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, mColour[1]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mZedCamera->getWidth(), mZedCamera->getHeight(), 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Create OpenGL images to view the depth stream
        glGenTextures(2, mDepth);
        glBindTexture(GL_TEXTURE_2D, mDepth[0]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, mRSCamera->getWidth(), mRSCamera->getHeight(), 0, GL_RED, GL_UNSIGNED_SHORT, nullptr));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, mDepth[1]);
        TEST_GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, mRSCamera->getWidth(), mRSCamera->getHeight(), 0, GL_RED, GL_UNSIGNED_SHORT, nullptr));
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
        cv::Mat frame, tmp;
        mZedCamera->capture();
        mRSCamera->capture();

        // Capture from chosen camera
        CameraSource* source = mShowColour ? (CameraSource*)mZedCamera : mRSCamera;

        // Display eyes
        mShader->bind();
        for (int i = 0; i < 2; i++)
        {
            // Capture frame
            source->copyFrameIntoCVImage((CameraSource::Eye)i, &frame);

            // Transform to be relative to the ZED
            if (!mShowColour)
            {
                tmp = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_16UC1);
                glm::mat3 rsCalib = convertCVToMat3<double>(mRSCamera->getCameraMatrix());
                glm::mat3 invRSCalib = glm::inverse(rsCalib);
                glm::mat3 rotation = convertCVToMat3<double>(mR);
                glm::vec3 translation(mT.at<double>(0), mT.at<double>(1), mT.at<double>(2));
                glm::mat3 zedCalib = convertCVToMat3<double>(mZedCamera->getCameraMatrix());
                /*
                cout << glm::to_string(rsCalib) << endl;
                cout << glm::to_string(invRSCalib) << endl;
                cout << glm::to_string(zedCalib) << endl;
                */
                // TODO: Instead of using depth_aligned_to_color, make use of the depth->colour extrinsic parameters from librealsense
                // TODO: Clean this code up
                for (int row = 0; row < frame.rows; row++)
                {
                    for (int col = 0; col < frame.cols; col++)
                    {
                        glm::vec3 point((float)col, (float)row, 1.0);
                        float depth = (float)frame.at<unsigned short>(row, col) * mRSCamera->getDepthScale();
                        point = (invRSCalib * point) * depth;
                        point = glm::inverse(rotation) * point;
                        point -= translation;
                        depth = point.z;
                        point.x /= depth;
                        point.y /= depth;
                        point.z = 1.0;
                        point = zedCalib * point;
                        if (point.x >= 0.0 && point.x <= frame.cols && point.y >= 0.0 && point.y <= frame.rows)
                            tmp.at<unsigned short>((int)point.y, (int)point.x) = (unsigned short)(depth / mRSCamera->getDepthScale());
                    }
                }
                tmp.copyTo(frame);
            }

            // Display
            if (mShowColour)
            {
                glBindTexture(GL_TEXTURE_2D, mColour[i]);
                TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, source->getWidth(), source->getHeight(), GL_BGR, GL_UNSIGNED_BYTE, frame.ptr()));
            }
            else
            {
                glBindTexture(GL_TEXTURE_2D, mDepth[i]);
                TEST_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, source->getWidth(), source->getHeight(), GL_RED, GL_UNSIGNED_SHORT, frame.ptr()));
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

private:
    bool mShowColour;

    ZEDCamera* mZedCamera;
    F200Camera* mRSCamera;

    Rectangle2D* mEyeQuad[2];
    Shader* mShader;
    GLuint mColour[2];
    GLuint mDepth[2];

    // Extrinsics for the camera pair
    cv::Mat mR, mT;
};

DEFINE_MAIN(ViewCalibration);
