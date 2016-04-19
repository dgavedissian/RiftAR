cmake_minimum_required(VERSION 3.0)

project(RiftAR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

# Dependencies
set(THIRD_PARTY_DIR "$ENV{THIRD_PARTY_DIR}")
set(OVR_SDK_ROOT ${THIRD_PARTY_DIR}/OculusSDK)
find_package(OpenGL)
find_package(OVR)
find_package(CUDA 7.5 REQUIRED)
find_package(OpenCV 2.4 COMPONENTS core highgui imgproc REQUIRED)

set(ZED_INCLUDE_DIRS $ENV{ZED_INCLUDE_DIRS})
set(ZED_LIBRARY_DIR $ENV{ZED_LIBRARY_DIR})
set(ZED_LIBRARIES $ENV{ZED_LIBRARIES_64})

set(INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/3rdparty # for GLM and GL3W
    ${OPENGL_INCLUDE_DIR}
    ${OVR_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${ZED_INCLUDE_DIRS})

set(LIBS
    ${OPENGL_LIBRARIES}
    ${OVR_LIBRARY}
    ${ZED_LIBRARIES})

# GLFW
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)
add_subdirectory(3rdparty/glfw)
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/3rdparty/glfw/include)
set(LIBS ${LIBS} glfw ${GLFW_LIBRARIES})

# librealsense
add_subdirectory(3rdparty/librealsense)
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/3rdparty/librealsense/include)
set(LIBS ${LIBS} librealsense)

# Add source files
set(SRC_FILES
    3rdparty/gl3w.c
    src/FullscreenQuad.cpp
    src/Shader.cpp
    # ---------------
    src/Main.cpp
    src/RealsenseF200.cpp)
set(HDR_FILES
    src/Common.h
    src/FullscreenQuad.h
    src/Shader.h
    # ---------------
    src/RealsenseF200.h)

link_directories(${ZED_LIBRARY_DIR})
add_executable(RiftAR ${SRC_FILES} ${HDR_FILES})
target_include_directories(RiftAR PRIVATE ${INCLUDE_DIRS})
target_link_libraries(RiftAR ${LIBS})
