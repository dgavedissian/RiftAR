cmake_minimum_required(VERSION 3.0)

project(RiftAR)

set(THIRD_PARTY_DIR "$ENV{THIRD_PARTY_DIR}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

# Dependencies
find_package(OpenGL)
find_package(CUDA 7.5 REQUIRED)
find_package(OpenCV 2.4 COMPONENTS core highgui calib3d REQUIRED)

set(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
set(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})
set(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})

set(INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/3rdparty # for GLM and GL3W
    ${OPENGL_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${ZED_INCLUDE_DIRS})

set(LIBS
    ${OPENGL_LIBRARIES}
	${CUDA_LIBRARIES}
	${OpenCV_LIBRARIES}
    ${ZED_LIBRARIES})

# GLFW
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)
add_subdirectory(3rdparty/glfw)
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/3rdparty/glfw/include)
set(LIBS ${LIBS} ${GLFW_LIBRARIES} glfw)

# librealsense
add_subdirectory(3rdparty/librealsense)
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/3rdparty/librealsense/include)
set(LIBS ${LIBS} librealsense)

# Add include directories
include_directories(${INCLUDE_DIRS})

# Common library
set(SRC_FILES
    3rdparty/gl3w.c
    src/lib/Main.cpp
    # GL helper objects ---------------
    src/lib/Rectangle2D.cpp
    src/lib/Shader.cpp
	src/lib/Model.cpp
    # Cameras -------------------------
    src/lib/CameraSource.cpp
    src/lib/F200Camera.cpp
    src/lib/ZEDCamera.cpp
    src/lib/ZEDCVCamera.cpp)
set(HDR_FILES
    src/lib/App.h
    src/lib/Common.h
    # GL helper objects ---------------
    src/lib/Rectangle2D.h
    src/lib/Shader.h
	src/lib/Model.h
    # Cameras -------------------------
    src/lib/CameraSource.h
    src/lib/F200Camera.h
    src/lib/ZEDCamera.h
    src/lib/ZEDCVCamera.h)
link_directories(${ZED_LIBRARY_DIR})
add_library(Common STATIC ${SRC_FILES} ${HDR_FILES})
target_link_libraries(Common ${LIBS})

# OVR
set(OVR_SDK_ROOT ${THIRD_PARTY_DIR}/OculusSDK)
find_package(OVR)

# kfusion
add_subdirectory(3rdparty/kfusion)

# RiftAR
set(RIFTAR_HDR
    src/DebugOutput.h
    src/KFusionTracker.h
    src/OutputContext.h
    src/RealsenseDepthAdjuster.h
    src/RenderContext.h
    src/RiftAR.h
    src/RiftOutput.h)
set(RIFTAR_SRC
    src/DebugOutput.cpp
    src/KFusionTracker.cu
    src/RealsenseDepthAdjuster.cu
    src/RenderContext.cpp
    src/RiftAR.cpp
    src/RiftOutput.cpp)
include_directories(${OVR_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/3rdparty/kfusion)
cuda_add_executable(RiftAR ${RIFTAR_SRC} ${RIFTAR_HDR})
target_link_libraries(RiftAR Common ${OVR_LIBRARY} kfusion)

# StereoCalibrate
add_executable(StereoCalibrate src/StereoCalibrate.cpp)
target_link_libraries(StereoCalibrate Common)
