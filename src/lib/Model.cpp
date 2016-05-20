#include "Common.h"
#include "Model.h"
#include "Shader.h"

#include <fstream>

Model::Model(const string& filename) :
    mVertexArrayObject(0),
    mVertexBufferObject(0),
    mShader(nullptr)
{
    cout << "Loading STL model: " << filename << endl;

    // Load data from the model file
    std::vector<glm::vec3> vertexData;
    std::ifstream in;
    in.open(filename, std::ios::binary | std::ios::in);
    load(in, vertexData);
    mVertexCount = (uint)vertexData.size() / 2;

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
    if (mShader)
        delete mShader;
}

void Model::setPosition(const glm::vec3& position)
{
    mTransform[3] = glm::vec4(position, 1.0f);
}

void Model::setOrientation(const glm::quat& orientation)
{
    glm::mat3 rotationMatrix = glm::mat3_cast(orientation);
    mTransform[0] = glm::vec4(rotationMatrix[0], 0.0f);
    mTransform[1] = glm::vec4(rotationMatrix[1], 0.0f);
    mTransform[2] = glm::vec4(rotationMatrix[2], 0.0f);
}

void Model::setTransform(const glm::mat4& transform)
{
    mTransform = transform;
}

void Model::render(const glm::mat4& view, const glm::mat4& projection)
{
    mShader->bind();
    mShader->setUniform("modelViewProjectionMatrix", projection * view * mTransform);
    mShader->setUniform("modelMatrix", mTransform);
    glBindVertexArray(mVertexArrayObject);
    glDrawArrays(GL_TRIANGLES, 0, mVertexCount);
}

const std::vector<glm::vec3>& Model::getVertices() const
{
    return mVertices;
}

glm::vec3 Model::getMin() const
{
    return mMin;
}

glm::vec3 Model::getMax() const
{
    return mMax;
}

glm::vec3 Model::getSize() const
{
     return mSize;
}

const glm::mat4& Model::getTransform() const
{
    return mTransform;
}

void Model::load(std::ifstream& in, std::vector<glm::vec3>& vertexData)
{
    uint32_t triangleCount;
    in.seekg(80, std::ios::beg);
    in.read((char*)&triangleCount, sizeof(uint32_t));

    vertexData.reserve(triangleCount * 3 * 2);
    mVertices.reserve(triangleCount * 3);
    cout << "- Triangles: " << triangleCount << endl;

    float inf = std::numeric_limits<float>::infinity();
    mMin = glm::vec3(inf);
    mMax = glm::vec3(-inf);

    for (int i = 0; i < (int)triangleCount; i++)
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
            
            // Transform point from millimetres to metres
            point *= 1e-3f;

            // Update bounding box
            if (point.x < mMin.x)
                mMin.x = point.x;
            if (point.y < mMin.y)
                mMin.y = point.y;
            if (point.z < mMin.z)
                mMin.z = point.z;
            if (point.x > mMax.x)
                mMax.x = point.x;
            if (point.y > mMax.y)
                mMax.y = point.y;
            if (point.z > mMax.z)
                mMax.z = point.z;

            vertexData.push_back(point);
            mVertices.push_back(point);
            vertexData.push_back(normal);
        }

        in.seekg(2, std::ios::cur);
    }

    mSize = mMax - mMin;

    cout << "min: " << glm::to_string(mMin) << endl;
    cout << "max: " << glm::to_string(mMax) << endl;

}
