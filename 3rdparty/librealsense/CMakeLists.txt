cmake_minimum_required(VERSION 3.0)

project(librealsense)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include(BasicConfig)

file(GLOB_RECURSE SRC "src/*")
file(GLOB_RECURSE HDR "include/librealsense/*")
list(APPEND SRC ${HDR})

if(WIN32)
	add_definitions(-DRS_USE_WMF_BACKEND -DUNICODE -D_UNICODE)
endif()
if(CMAKE_COMPILER_IS_GNUCXX)
	add_definitions(-DRS_USE_LIBUVC_BACKEND -DUNICODE -D_UNICODE)
endif()
add_library(${PROJECT_NAME} STATIC ${SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_PDB_NAME_DEBUG "${PROJECT_NAME}${CMAKE_DEBUG_POSTFIX}")

target_include_directories(${PROJECT_NAME} PUBLIC "include")
