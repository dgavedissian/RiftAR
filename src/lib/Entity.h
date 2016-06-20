#pragma once

class Model;
class Shader;

class Entity
{
public:
    Entity(shared_ptr<Model> m, shared_ptr<Shader> s);
    ~Entity();

    void setPosition(const glm::vec3& position);
    void setOrientation(const glm::quat& orientation);
    void setTransform(const glm::mat4& transform);

    void setModel(shared_ptr<Model> model);
    void setShader(shared_ptr<Shader> shader);

    void render(const glm::mat4& view, const glm::mat4& projection);

    shared_ptr<Model> getModel();
    shared_ptr<Shader> getShader();
    const glm::mat4& getTransform() const;

private:
    shared_ptr<Model> mModel;
    shared_ptr<Shader> mShader;
    glm::mat4 mTransform;
};