cmake_minimum_required(VERSION 3.0)

project(RiftAR)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

# Dependencies
set(THIRD_PARTY_DIR "$ENV{THIRD_PARTY_DIR}")
set(GLFW_LOCATION ${THIRD_PARTY_DIR}/glfw)
find_package(OpenGL)
find_package(GLFW 3)
set(OVR_SDK_ROOT ${THIRD_PARTY_DIR}/OculusSDK)
find_package(OVR)
set(INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/3rdparty # for GLM and GL3W
    ${OPENGL_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIR}
    ${OVR_INCLUDE_DIR})
set(LIBS
    ${OPENGL_LIBRARIES}
    ${GLFW_LIBRARIES}
    ${OVR_LIBRARY})

# Librealsense
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

add_executable(RiftAR ${SRC_FILES} ${HDR_FILES})
target_include_directories(RiftAR PRIVATE ${INCLUDE_DIRS})
target_link_libraries(RiftAR ${LIBS})
