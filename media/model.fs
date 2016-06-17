#version 330

uniform vec4 diffuseColour;

in vec3 oNormal;

out vec4 colour;

void main()
{
    float lighting = clamp(dot(oNormal, vec3(1.0, 0.0, 0.0)), 0.0, 1.0);
    lighting = max(lighting, 0.3); // ambient factor
    colour = vec4(lighting, lighting, lighting, 1.0) * diffuseColour;
}