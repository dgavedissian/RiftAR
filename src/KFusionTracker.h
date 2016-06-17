#pragma once

#include <kfusion/kfusion.h>

#include "simplex/DropSimplex.h"

class Entity;
class Model;
class RealsenseCamera;

class KFusionTracker
{
public:
    KFusionTracker(RealsenseCamera* camera);
    ~KFusionTracker();

    bool update(cv::Mat frame);

    void beginSearchingFor(Entity* target);
    bool checkTargetPosition(glm::mat4& resultTransform);
    bool isSearching() const;

    void reset();

    glm::mat4 getCameraPose() const;

    void getCurrentView(cv::Mat& mat);

private:
    float getCost(Model* model, Volume volume, const glm::mat4& transform);

    glm::mat3 convKFusionCoordSystem(const glm::mat3& rotation) const;
    glm::mat4 convKFusionCoordSystem(const glm::mat4& transform) const;

    // Cost Function
    class CostFunction : public drop::CostFunctionSimplex
    {
    public:
        CostFunction(Model* model, Volume volume, KFusionTracker* tracker);

        virtual double evaluate(const std::vector<double> &parameters);

        static glm::mat4 mat4FromParameters(const std::vector<double>& parameters);
        static void mat4ToParameters(const glm::mat4& matrix, std::vector<double>& parameters);

    private:
        Model* mModel;
        Volume mVolume;
        KFusionTracker* mTracker;

    };

    // KFusion
    RealsenseCamera* mSource;
    KFusion* mKFusion;
    Image<uint16_t, HostDevice> mDepthImage;
    glm::vec3 mNewOrigin;
    drop::SimplexOptimizer mOptimiser;

    // Initialisation parameters
    bool mIsInitialising;
    int mFrameCounter;

    // Current pose
    glm::mat4 mCameraPose;

    // Searching
    Entity* mSearchTarget; // when this is nullptr, we are not searching for anything

    // Debug Image
    Image<uchar4, HostDevice> mLightModel;
};
