#version 330

uniform sampler2D rgbCameraImage;
uniform sampler2D depthCameraImage;

in vec2 oTexCoord;

out vec4 colour;

void main()
{
    gl_FragDepth = texture(depthCameraImage, oTexCoord).r;
    colour = vec4(texture(rgbCameraImage, oTexCoord).bgr, 1.0);
}