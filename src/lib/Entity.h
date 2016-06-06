#pragma once

class Model;
class Shader;

class Entity
{
public:
    Entity(Model* m, Shader* s);
    ~Entity();

    void setPosition(const glm::vec3& position);
    void setOrientation(const glm::quat& orientation);
    void setTransform(const glm::mat4& transform);

    void setModel(Model* model);
    void setShader(Shader* shader);

    void render(const glm::mat4& view, const glm::mat4& projection);

    Model* getModel();
    Shader* getShader();
    const glm::mat4& getTransform() const;

    static Entity* loadModel(const string& filename);

private:
    Model* mModel;
    Shader* mShader;
    glm::mat4 mTransform;
};