#pragma once

class Shader;

class Model
{
public:
    Model(const string& filename);
    ~Model();

    void setPosition(const glm::vec3& position);
    void setOrientation(const glm::quat& orientation);
    void setTransform(const glm::mat4& transform);

    void render(const glm::mat4& view, const glm::mat4& projection);

    const std::vector<glm::vec3>& getVertices() const;

    glm::vec3 getMin() const;
    glm::vec3 getMax() const;
    glm::vec3 getSize() const;

    const glm::mat4& getTransform() const;


private:
    void load(std::ifstream& in, std::vector<glm::vec3>& vertexData);

    std::vector<glm::vec3> mVertices;

    GLuint mVertexArrayObject;
    GLuint mVertexBufferObject;
    uint mVertexCount;

    Shader* mShader;
    glm::mat4 mTransform;

    glm::vec3 mMin, mMax, mSize;

};