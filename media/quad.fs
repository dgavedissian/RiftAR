#version 330

uniform sampler2D cameraImage;

in vec2 oTexCoord;

out vec4 colour;

void main()
{
    colour = vec4(texture(cameraImage, oTexCoord).rgb, 1.0);
}