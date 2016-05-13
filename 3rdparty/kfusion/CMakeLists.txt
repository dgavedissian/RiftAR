cmake_minimum_required(VERSION 3.0)

project(kfusion)

set(KFUSION_SRC
    helpers.cu
    kfusion.cu)

set(KFUSION_HDR
    helpers.h
    kfusion.h
    cutil_math.h)

find_package(CUDA 7.5 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/3rdparty)
cuda_add_library(kfusion STATIC ${KFUSION_SRC} ${KFUSION_HDR})
