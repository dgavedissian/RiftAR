#version 330

uniform sampler2D rgbCameraImage;
uniform sampler2D depthCameraImage;
uniform bool invertColour;

in vec2 oTexCoord;

out vec4 colour;

uniform float znear, zfar;
uniform float depthScale;

void main()
{
    float depth = texture(depthCameraImage, oTexCoord).r * depthScale;
    float projectedDepth = (zfar + znear) / (zfar - znear) - 2.0 * znear * zfar / (depth * (zfar - znear));
    gl_FragDepth = 0.5 * projectedDepth + 0.5;
    if (invertColour)
    {
        colour = vec4(texture(rgbCameraImage, oTexCoord).bgr, 1.0);
    }
    else
    {
        colour = vec4(texture(rgbCameraImage, oTexCoord).rgb, 1.0);
    }
}