#version 330

uniform sampler2D cameraImage;
uniform bool invertColour;

in vec2 oTexCoord;

out vec4 colour;

void main()
{
    if (invertColour)
    {
        colour = vec4(texture(cameraImage, oTexCoord).bgr, 1.0);
    }
    else
    {
        colour = vec4(texture(cameraImage, oTexCoord).rgb, 1.0);
    }
}