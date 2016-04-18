#include "Common.h"
#include "FullscreenQuad.h"

FullscreenQuad::FullscreenQuad() :
    mVertexArrayObject(0),
    mVertexBufferObject(0),
    mElementBufferObject(0)
{
    // Declare vertex and index data
    float vertexData[] = {
        -1.0f, 1.0f,    // top left
        1.0f, 1.0f,     // top right
        -1.0f, -1.0f,   // bottom left
        1.0f, -1.0f     // bottom right
    };
    GLuint elementData[] = {
        0, 1, 2, 2, 1, 3
    };

    // Create vertex array object to hold buffers
    glGenVertexArrays(1, &mVertexArrayObject);
    glBindVertexArray(mVertexArrayObject);

    // Generate vertex buffer
    glGenBuffers(1, &mVertexBufferObject);
    glBindBuffer(GL_ARRAY_BUFFER, mVertexBufferObject);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertexData), vertexData, GL_STATIC_DRAW);

    // Set up vertex layout
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);

    // Set up element buffer
    glGenBuffers(1, &mElementBufferObject);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mElementBufferObject);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elementData), elementData, GL_STATIC_DRAW);
}

FullscreenQuad::~FullscreenQuad()
{

}

void FullscreenQuad::render()
{
    glBindVertexArray(mVertexArrayObject);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}
