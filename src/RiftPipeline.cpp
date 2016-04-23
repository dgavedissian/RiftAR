#include "Common.h"
#include "RiftPipeline.h"

#include "StereoCamera.h"
#include "Rectangle2D.h"
#include "Shader.h"

RiftPipeline::RiftPipeline()
{
    ovrResult result = ovr_Initialize(nullptr);
    if (OVR_FAILURE(result))
        throw std::runtime_error("Failed to initialise LibOVR");

    // Create a context for the rift device
    result = ovr_Create(&mSession, &mLuid);
    if (OVR_FAILURE(result))
        throw std::runtime_error("Oculus Rift not detected");
}

RiftPipeline::~RiftPipeline()
{
    ovr_Destroy(mSession);
    ovr_Shutdown();
}

void RiftPipeline::display(StereoCamera* input)
{
    static Rectangle2D leftQuad(glm::vec2(0.0f, 0.0f), glm::vec2(0.5f, 1.0f));
    static Rectangle2D rightQuad(glm::vec2(0.5f, 0.0f), glm::vec2(1.0f, 1.0f));
    static Shader shader("../media/fullscreenquad.vs", "../media/fullscreenquad.fs");

    shader.bind();
    input->retrieve();
    glBindTextureUnit(0, input->getTexture(StereoCamera::LEFT));
    leftQuad.render();
    glBindTextureUnit(0, input->getTexture(StereoCamera::RIGHT));
    rightQuad.render();
}
