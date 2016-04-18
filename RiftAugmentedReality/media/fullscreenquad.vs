#version 330

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 texCoord;

out vec2 oTexCoord;

void main()
{
    gl_Position = vec4(position, 1.0, 1.0);

    // Flipping the T axis of the texture coordinates is required as the camera stream assumes a
    // coordinate system where the top left is (0,0)
    oTexCoord = vec2(position.x, -position.y) * 0.5 + 0.5;
}