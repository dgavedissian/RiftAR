#pragma once

class Shader;

class Model
{
public:
    Model(const string& filename);
    ~Model();

    void setPosition(const glm::vec3& position);
    void setOrientation(const glm::quat& orientation);

    void render(const glm::mat4& view, const glm::mat4& projection);

    glm::vec3 getMin() const;
    glm::vec3 getMax() const;
    glm::vec3 getSize() const;

private:
    void load(std::ifstream& in, std::vector<glm::vec3>& vertexData);

    GLuint mVertexArrayObject;
    GLuint mVertexBufferObject;
    uint mVertexCount;

    Shader* mShader;
    glm::mat4 mModelMatrix;

    glm::vec3 mMin, mMax, mSize;

};