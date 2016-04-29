#version 330

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 texCoord;

out vec2 oTexCoord;

void main()
{
    vec2 normalisedDeviceCoords = position * 2.0 - 1.0;
    gl_Position = vec4(normalisedDeviceCoords.x, -normalisedDeviceCoords.y, 1.0, 1.0);

    // Flipping the T axis of the texture coordinates is required as the camera stream assumes a
    // coordinate system where the top left is (0,0)
    oTexCoord = vec2(texCoord.x, -texCoord.y);
}