#pragma once

class STLModel
{
public:
    STLModel(const string& filename);
    ~STLModel();

    void render();

private:
    void load(std::ifstream& in, std::vector<glm::vec3>& vertexData);

    GLuint mVertexArrayObject;
    GLuint mVertexBufferObject;
    uint mVertexCount;

};