#include "Common.h"
#include "Entity.h"
#include "Model.h"
#include "Shader.h"

Entity::Entity(Model* m, Shader* s) :
    mModel(m),
    mShader(s)
{
}

Entity::~Entity()
{
}

void Entity::setPosition(const glm::vec3& position)
{
    mTransform[3] = glm::vec4(position, 1.0f);
}

void Entity::setOrientation(const glm::quat& orientation)
{
    glm::mat3 rotationMatrix = glm::mat3_cast(orientation);
    mTransform[0] = glm::vec4(rotationMatrix[0], 0.0f);
    mTransform[1] = glm::vec4(rotationMatrix[1], 0.0f);
    mTransform[2] = glm::vec4(rotationMatrix[2], 0.0f);
}

void Entity::setTransform(const glm::mat4& transform)
{
    mTransform = transform;
}

void Entity::setModel(Model* model)
{
    mModel = model;
}

void Entity::setShader(Shader* shader)
{
    mShader = shader;
}

void Entity::render(const glm::mat4& view, const glm::mat4& projection)
{
    mShader->bind();
    mShader->setUniform("modelViewProjectionMatrix", projection * view * mTransform);
    mShader->setUniform("modelMatrix", mTransform);
    mModel->render();
}

Model* Entity::getModel()
{
    return mModel;
}

Shader* Entity::getShader()
{
    return mShader;
}

const glm::mat4& Entity::getTransform() const
{
    return mTransform;
}

Entity* Entity::loadModel(const string& filename)
{
    return new Entity(new Model(filename), new Shader("../media/model.vs", "../media/model.fs"));
}
