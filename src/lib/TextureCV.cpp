#include "Common.h"
#include "TextureCV.h"

TextureCV::TextureCV()
{
    glGenTextures(1, &mTexture);
    glBindTexture(GL_TEXTURE_2D, mTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

TextureCV::TextureCV(cv::Mat mat) :
    TextureCV()
{
    mMat = mat;
    mSize = cv::Size(mMat.cols, mMat.rows);

    // We can assume the texture is already bound from the delegated constructor
    if (mSize.width > 0 && mSize.height > 0)
    {
        GL_CHECK(glTexImage2D(
            GL_TEXTURE_2D, 0,
            getInternalFormat(),
            mSize.width, mSize.height,
            0,
            getFormat(),
            getType(),
            mMat.data
            ));
    }
}

void TextureCV::update()
{
    glBindTexture(GL_TEXTURE_2D, mTexture);
    if (mSize.width != mMat.cols || mSize.height != mMat.rows)
    {
        mSize = cv::Size(mMat.cols, mMat.rows);
        if (mSize.width > 0 && mSize.height > 0)
        {
            GL_CHECK(glTexImage2D(
                GL_TEXTURE_2D, 0,
                getInternalFormat(),
                mSize.width, mSize.height,
                0,
                getFormat(),
                getType(),
                mMat.data
                ));
        }
    }
    else
    {
        GL_CHECK(glTexSubImage2D(
            GL_TEXTURE_2D, 0,
            0, 0,
            mSize.width, mSize.height,
            getFormat(),
            getType(),
            mMat.data
            ));
    }
}

cv::Mat& TextureCV::getCVMat()
{
    return mMat;
}

GLuint TextureCV::getGLTexture()
{
    return mTexture;
}

GLint TextureCV::getInternalFormat()
{
    static std::map<int, GLint> internalFormatMapper;
    if (internalFormatMapper.empty())
    {
        internalFormatMapper[CV_8UC1] = GL_R8;
        internalFormatMapper[CV_8UC2] = GL_RG8;
        internalFormatMapper[CV_8UC3] = GL_RGB8;
        internalFormatMapper[CV_8UC4] = GL_RGBA8;
        internalFormatMapper[CV_8SC1] = GL_R8I;
        internalFormatMapper[CV_8SC2] = GL_RG8I;
        internalFormatMapper[CV_8SC3] = GL_RGB8I;
        internalFormatMapper[CV_8SC4] = GL_RGBA8I;
        internalFormatMapper[CV_16UC1] = GL_R16;
        internalFormatMapper[CV_16UC2] = GL_RG16;
        internalFormatMapper[CV_16UC3] = GL_RGB16;
        internalFormatMapper[CV_16UC4] = GL_RGBA16;
        internalFormatMapper[CV_16SC1] = GL_R16I;
        internalFormatMapper[CV_16SC2] = GL_RG16I;
        internalFormatMapper[CV_16SC3] = GL_RGB16I;
        internalFormatMapper[CV_16SC4] = GL_RGBA16I;
        internalFormatMapper[CV_32SC1] = GL_R32I;
        internalFormatMapper[CV_32SC2] = GL_RG32I;
        internalFormatMapper[CV_32SC3] = GL_RGB32I;
        internalFormatMapper[CV_32SC4] = GL_RGBA32I;
        internalFormatMapper[CV_32FC1] = GL_R32F;
        internalFormatMapper[CV_32FC2] = GL_RG32F;
        internalFormatMapper[CV_32FC3] = GL_RGB32F;
        internalFormatMapper[CV_32FC4] = GL_RGBA32F;
    }
    auto it = internalFormatMapper.find(mMat.type());
    if (it == internalFormatMapper.end())
        THROW_ERROR("Unsupported cv::Mat format");
    return it->second;
}

GLenum TextureCV::getFormat()
{
    // Use BGR/BGRA due to OpenCV convention
    switch (mMat.channels())
    {
    case 1: return GL_RED;
    case 2: return GL_RG;
    case 3: return GL_BGR;
    case 4: return GL_BGRA;
    default: THROW_ERROR("Invalid number of channels");
    }
}

GLenum TextureCV::getType()
{
    switch (mMat.depth())
    {
    case CV_8U: return GL_UNSIGNED_BYTE;
    case CV_8S: return GL_BYTE;
    case CV_16U: return GL_UNSIGNED_SHORT;
    case CV_16S: return GL_SHORT;
    case CV_32S: return GL_UNSIGNED_INT;
    case CV_32F: return GL_FLOAT;
    default: THROW_ERROR("Unsupported cv::Mat type");
    }
}
