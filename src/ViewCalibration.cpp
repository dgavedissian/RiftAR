#include "lib/Common.h"
#include "lib/F200Camera.h"
#include "lib/ZEDCamera.h"
#include "lib/Rectangle2D.h"
#include "lib/Shader.h"
#include "lib/STLModel.h"

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
        mRSColourToZedLeft = buildExtrinsic(glm::inverse(convertCVToMat3<double>(R)), -convertCVToVec3<double>(T));

        // Create objects
        mQuad = new Rectangle2D(glm::vec2(0.0f, 0.0f), glm::vec2(1.0f, 1.0f));
        mFullscreenShader = new Shader("../media/quad.vs", "../media/quad_inv.fs");
        mFullscreenWithDepthShader = new Shader("../media/quad.vs", "../media/quad_inv_depth.fs");
        mFullscreenWithDepthShader->bind();
        mFullscreenWithDepthShader->setUniform<int>("rgbCameraImage", 0);
        mFullscreenWithDepthShader->setUniform<int>("depthCameraImage", 1);

        glm::mat4 model = glm::translate(glm::mat4(), glm::vec3(-0.1f, -0.1f, -0.4f));
        glm::mat4 view;
        glm::mat4 projection = glm::perspective(glm::radians(75.0f), 640.0f / 480.0f, 0.01f, 10.0f);
        mModel = new STLModel("../media/meshes/skull.stl");
        mModelShader = new Shader("../media/model.vs", "../media/model.fs");
        mModelShader->bind();
        mModelShader->setUniform("modelViewProjectionMatrix", projection * model);
        mModelShader->setUniform("modelMatrix", model);

        // Enable culling and depth testing
        glEnable(GL_CULL_FACE);
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
        glClear(GL_DEPTH_BUFFER_BIT);

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
            glm::mat4 depthToColour = mRealsense->getExtrinsics(F200Camera::DEPTH, F200Camera::COLOUR);

            // Combined extrinsics mapping RS depth to ZED
            glm::mat4 rsToZed = mRSColourToZedLeft * depthToColour;
            if (i == 1) // Right eye
            {
                rsToZed = mZed->getExtrinsics(ZEDCamera::LEFT, ZEDCamera::RIGHT) * rsToZed;
            }

            // TODO: Clean this code up, and port to CUDA?
            glm::vec3 point2d;
            glm::vec4 point3d;
            for (int row = 0; row < frame.rows; row++)
            {
                for (int col = 0; col < frame.cols; col++)
                {
                    // Read depth
                    unsigned short depthPixel = frame.at<unsigned short>(row, col);
                    if (depthPixel == 0)
                        continue;
                    float depth = (float)depthPixel * mRealsense->getDepthScale();
                    float newDepth;

                    // Top left of depth pixel in homogenous coordinates (xd,yd,d)
                    point2d = glm::vec3((float)col - 0.5f, (float)row - 0.5f, 1.0f) * depth;
                    // Deproject pixel to point and convert to 4D homogenous coordinates
                    point3d = glm::vec4(invRSCalib * point2d, 1.0);
                    // Map from Depth -> ZED
                    point3d = rsToZed * point3d;
                    // Project point - conversion from vec3 to vec3 is equiv to multiplying by [I|0] matrix
                    point2d = zedCalib * glm::vec3(point3d.x, point3d.y, point3d.z);
                    // Record depth and convert to cartesian
                    newDepth = point2d.z;
                    point2d /= point2d.z;
                    cv::Point start((int)std::round(point2d.x), (int)std::round(point2d.y));

                    // Bottom right of depth pixel in homogenous coordinates (xd,yd,d)
                    point2d = glm::vec3((float)col + 0.5f, (float)row + 0.5f, 1.0f) * depth;
                    // Deproject pixel to point and convert to 4D homogenous coordinates
                    point3d = glm::vec4(invRSCalib * point2d, 1.0);
                    // Map from Depth -> ZED
                    point3d = rsToZed * point3d;
                    // Project point - conversion from vec3 to vec3 is equiv to multiplying by [I|0] matrix
                    point2d = zedCalib * glm::vec3(point3d.x, point3d.y, point3d.z);
                    // Record depth and convert to cartesian
                    newDepth = point2d.z;
                    point2d /= point2d.z;
                    cv::Point end((int)std::round(point2d.x), (int)std::round(point2d.y));

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
                            writeDepth(transformedDepth, x, y, newDepth);
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
            glDisable(GL_DEPTH_TEST);
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

                // Display the mesh
                glEnable(GL_DEPTH_TEST);
                mModelShader->bind();
                mModel->render();
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
        return cv::Size(1600, 600);
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

    STLModel* mModel;
    Shader* mModelShader;

    // Extrinsics for the camera pair
    glm::mat4 mRSColourToZedLeft;
};

DEFINE_MAIN(ViewCalibration);
