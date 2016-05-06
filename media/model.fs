#version 330

in vec3 oNormal;

out vec4 colour;

void main()
{
    float lighting = clamp(dot(oNormal, vec3(1.0, 0.0, 0.0)), 0.0, 1.0);
    colour = vec4(lighting, lighting, lighting, 1.0);
}