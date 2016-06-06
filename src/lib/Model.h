#pragma once

class Shader;

class Model
{
public:
    Model(const string& filename);
    ~Model();

    void render();

    const std::vector<glm::vec3>& getVertices() const;

    glm::vec3 getMin() const;
    glm::vec3 getMax() const;
    glm::vec3 getSize() const;

private:
    void load(std::ifstream& in, std::vector<glm::vec3>& vertexData);

    std::vector<glm::vec3> mVertices;

    GLuint mVertexArrayObject;
    GLuint mVertexBufferObject;
    uint mVertexCount;

    glm::vec3 mMin, mMax, mSize;

};