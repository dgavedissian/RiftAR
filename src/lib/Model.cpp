#include "Common.h"
#include "Model.h"
#include "Shader.h"

#include <fstream>

Model::Model(const string& filename) :
    mVertexArrayObject(0),
    mVertexBufferObject(0)
{
    cout << "Loading STL model: " << filename << endl;

    // Load data from the model file
    std::vector<glm::vec3> vertexData;
    std::ifstream in;
    in.open(filename, std::ios::binary | std::ios::in);
    load(in, vertexData);
    mVertexCount = vertexData.size() / 2;

    // Create vertex array object to hold buffers
    glGenVertexArrays(1, &mVertexArrayObject);
    glBindVertexArray(mVertexArrayObject);

    // Generate vertex buffer
    glGenBuffers(1, &mVertexBufferObject);
    glBindBuffer(GL_ARRAY_BUFFER, mVertexBufferObject);
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(glm::vec3), vertexData.data(), GL_STATIC_DRAW);

    // Set up vertex layout
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    // Load the shader
    mShader = new Shader("../media/model.vs", "../media/model.fs");
}

Model::~Model()
{
    delete mShader;
}

void Model::setPosition(const glm::vec3& position)
{
    mModelMatrix[3] = glm::vec4(position, 1.0f);
}

void Model::setOrientation(const glm::quat& orientation)
{
    glm::mat3 rotationMatrix = glm::mat3_cast(orientation);
    mModelMatrix[0] = glm::vec4(rotationMatrix[0], 0.0f);
    mModelMatrix[1] = glm::vec4(rotationMatrix[1], 0.0f);
    mModelMatrix[2] = glm::vec4(rotationMatrix[2], 0.0f);
}

void Model::render(const glm::mat4& view, const glm::mat4& projection)
{
    mShader->bind();
    mShader->setUniform("modelViewProjectionMatrix", projection * view * mModelMatrix);
    mShader->setUniform("modelMatrix", mModelMatrix);
    glBindVertexArray(mVertexArrayObject);
    glDrawArrays(GL_TRIANGLES, 0, mVertexCount);
}

void Model::load(std::ifstream& in, std::vector<glm::vec3>& vertexData)
{
    uint32_t triangleCount;
    in.seekg(80, std::ios::beg);
    in.read((char*)&triangleCount, sizeof(uint32_t));

    vertexData.reserve(triangleCount * 6 * 3);
    cout << "- Triangles: " << triangleCount << endl;

    float inf = std::numeric_limits<float>::infinity();

    for (int i = 0; i < triangleCount; i++)
    {
        glm::vec3 normal;
        in.read((char*)&normal.x, sizeof(float));
        in.read((char*)&normal.y, sizeof(float));
        in.read((char*)&normal.z, sizeof(float));

        for (int j = 0; j < 3; j++)
        {
            glm::vec3 point;
            in.read((char*)&point.x, sizeof(float));
            in.read((char*)&point.y, sizeof(float));
            in.read((char*)&point.z, sizeof(float));
            
            // Scale point to metres scale from mm
            point *= 1e-3f;

            vertexData.push_back(point);
            vertexData.push_back(normal);
        }

        in.seekg(2, std::ios::cur);
    }

}
