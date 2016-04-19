#pragma once

// Full screen rectangle. Used for drawing a texture to the backbuffer directly, or for post processing effects.
class FullscreenQuad
{
public:
    FullscreenQuad();
    ~FullscreenQuad();

    void render();

private:
    GLuint mVertexArrayObject;
    GLuint mVertexBufferObject;
    GLuint mElementBufferObject;

};