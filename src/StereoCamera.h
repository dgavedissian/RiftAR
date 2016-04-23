#pragma once

class StereoCamera
{
public:
    enum Eye
    {
        LEFT = 0,
        RIGHT = 1
    };

    StereoCamera();
    virtual ~StereoCamera();

    virtual void retrieve() = 0;
    GLuint getTexture(StereoCamera::Eye e);

protected:
    GLuint mTexture[2];

};