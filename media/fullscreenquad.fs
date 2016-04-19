#version 330

uniform sampler2D cameraImage;

in vec2 oTexCoord;

out vec4 colour;

void main()
{
    colour = vec4(texture(cameraImage, oTexCoord).r, 0.0, 0.0, 1.0);
}