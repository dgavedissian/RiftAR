#pragma once

#include <librealsense/rs.hpp>

#include "CameraSource.h"

class F200Camera : public CameraSource
{
public:
    enum Stream
    {
        COLOUR = 1,
        DEPTH = 2,
        INFRARED = 4,
        INFRARED2 = 8
    };

    F200Camera(int width, int height, int frameRate, int streams);
    ~F200Camera();

    void setStream(Stream stream);

    void capture() override;
    void updateTextures() override;
    void copyFrameIntoCudaImage(Eye e, cudaGraphicsResource* resource) override;
    void copyFrameIntoCVImage(Eye e, cv::Mat* mat) override;
    const void* getRawData(Eye e) override;

private:
    Stream mCurrentStream;
    rs::context* mContext;
    rs::device* mDevice;

    GLuint mStreamTextures[4]; // one for each stream
};