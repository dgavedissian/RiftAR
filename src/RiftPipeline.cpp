#include "Common.h"
#include "RiftPipeline.h"

#include "CameraSource.h"
#include "Rectangle2D.h"
#include "Shader.h"

//#define USE_OCULUS

RiftPipeline::RiftPipeline()
{
#ifdef USE_OCULUS
    ovrResult result = ovr_Initialize(nullptr);
    if (OVR_FAILURE(result))
        throw std::runtime_error("Failed to initialise LibOVR");

    // Create a context for the rift device
    result = ovr_Create(&mSession, &mLuid);
    if (OVR_FAILURE(result))
        throw std::runtime_error("Oculus Rift not detected");
#endif
}

RiftPipeline::~RiftPipeline()
{
#ifdef USE_OCULUS
    ovr_Destroy(mSession);
    ovr_Shutdown();
#endif
}

void RiftPipeline::display(CameraSource* source)
{
#ifdef USE_OCULUS
    // TODO
#else
    static Rectangle2D leftQuad(glm::vec2(0.0f, 0.0f), glm::vec2(0.5f, 1.0f));
    static Rectangle2D rightQuad(glm::vec2(0.5f, 0.0f), glm::vec2(1.0f, 1.0f));
    static Shader shader("../media/fullscreenquad.vs", "../media/fullscreenquad.fs");

    shader.bind();
    source->capture();
    source->updateTextures();
    glBindTextureUnit(0, source->getTexture(CameraSource::LEFT));
    leftQuad.render();
    glBindTextureUnit(0, source->getTexture(CameraSource::RIGHT));
    rightQuad.render();
#endif
}
