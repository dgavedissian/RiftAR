#include "Common.h"
#include "Rectangle2D.h"

Rectangle2D::Rectangle2D(const glm::vec2& begin, const glm::vec2& end) :
    mVertexArrayObject(0),
    mVertexBufferObject(0),
    mElementBufferObject(0)
{
    // Declare vertex and index data
    float vertexData[] = {
        begin.x, begin.y,  0.0f, 1.0f,  // top left
        end.x, begin.y,    1.0f, 1.0f,  // top right
        begin.x, end.y,    0.0f, 0.0f,  // bottom left
        end.x, end.y,      1.0f, 0.0f   // bottom right
    };
    GLuint elementData[] = {
        2, 1, 0, 3, 1, 2
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
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    // Set up element buffer
    glGenBuffers(1, &mElementBufferObject);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mElementBufferObject);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elementData), elementData, GL_STATIC_DRAW);
}

Rectangle2D::~Rectangle2D()
{

}

void Rectangle2D::render()
{
    glBindVertexArray(mVertexArrayObject);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}
