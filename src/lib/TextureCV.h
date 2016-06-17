#pragma once

class TextureCV
{
public:
    TextureCV();
    TextureCV(cv::Mat mat);

    void update();

    cv::Mat& getCVMat();
    GLuint getGLTexture();

private:
    cv::Mat mMat;
    GLuint mTexture;

    cv::Size mSize;

    GLint getInternalFormat();
    GLenum getFormat();
    GLenum getType();

};