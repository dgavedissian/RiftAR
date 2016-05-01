cmake_minimum_required(VERSION 3.0)

project(RiftAR)

set(THIRD_PARTY_DIR "$ENV{THIRD_PARTY_DIR}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

# Dependencies
find_package(OpenGL)
find_package(CUDA 7.5 REQUIRED)
find_package(OpenCV 2.4 COMPONENTS core highgui imgproc calib3d REQUIRED)

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

# OVR
set(OVR_SDK_ROOT ${THIRD_PARTY_DIR}/OculusSDK)
find_package(OVR)

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

# Add a new application
function(add_app NAME SRC)
    set(SRC_FILES
        ${SRC}
        3rdparty/gl3w.c
        src/Main.cpp
        # GL helper objects ---------------
        src/Rectangle2D.cpp
        src/Shader.cpp
        # Camera ---------------
        src/F200Camera.cpp
        src/ZEDCamera.cpp
        src/ZEDOpenCVCamera.cpp)
    set(HDR_FILES
        src/App.h
        src/Common.h
        # GL helper objects ---------------
        src/Rectangle2D.h
        src/Shader.h
        # Camera ---------------
        src/CameraSource.h
        src/F200Camera.h
        src/ZEDCamera.h
        src/ZEDOpenCVCamera.h)

    link_directories(${ZED_LIBRARY_DIR})
    add_executable(${NAME} ${SRC_FILES} ${HDR_FILES})
    target_include_directories(${NAME} PRIVATE ${INCLUDE_DIRS})
    target_link_libraries(${NAME} ${LIBS})
endfunction()

# RiftAR
add_app(RiftAR src/RiftAR.cpp)
target_include_directories(RiftAR PRIVATE ${OVR_INCLUDE_DIR})
target_link_libraries(RiftAR ${OVR_LIBRARY})

# StereoCalibrate
add_app(StereoCalibrate src/StereoCalibrate.cpp)

# ViewCalibration
add_app(ViewCalibration src/ViewCalibration.cpp)