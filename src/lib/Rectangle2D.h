#pragma once

// Full screen rectangle. Used for drawing a texture to the backbuffer directly, or for post processing effects.
class Rectangle2D
{
public:
    Rectangle2D(const glm::vec2& begin, const glm::vec2& end);
    ~Rectangle2D();

    void render();

private:
    GLuint mVertexArrayObject;
    GLuint mVertexBufferObject;
    GLuint mElementBufferObject;

};