# - Find the Oculus Rift SDK
# This module defines the following variables:
#  OVR_INCLUDE_DIRS - include directories for OVR
#  OVR_LIBRARIES - libraries to link against OVR
#  OVR_FOUND - true if OVR has been found and can be used
#  OVR_DEFINITIONS - defines OVR_ENABLED

find_path(OVR_INCLUDE_DIR OVR_CAPI.h
    PATH_SUFFIXES "LibOVR/Include"
    PATHS ${OVR_SDK_ROOT})

if(WIN32)
    if(MSVC10)
        set(OVR_MSVC "VS2010")
    elseif(MSVC11)
        set(OVR_MSVC "VS2012")
    elseif(MSVC12)
        set(OVR_MSVC "VS2013")
    elseif(MSVC14)
        set(OVR_MSVC "VS2015")
    endif()
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(WIN32)
        set(OVR_PATH_SUFFIX "LibOVR/Lib/Windows/x64/Release/${OVR_MSVC}")
    elseif(APPLE)
        set(OVR_PATH_SUFFIX "LibOVR/Lib/Mac/Release")
    else()
        set(OVR_PATH_SUFFIX "LibOVR/Lib/Linux/Release/x86_64")
    endif()
else()
    if(WIN32)
        set(OVR_PATH_SUFFIX "LibOVR/Lib/Windows/Win32/Release/${OVR_MSVC}")
    else()
        set(OVR_PATH_SUFFIX "LibOVR/Lib/Linux/Release/i386")
    endif()
endif()

find_library(OVR_LIBRARY
    NAMES OVR ovr LibOVR libovr
    PATH_SUFFIXES ${OVR_PATH_SUFFIX}
    PATHS ${OVR_SDK_ROOT})

set(OVR_INCLUDE_DIRS ${OVR_INCLUDE_DIR})
set(OVR_LIBRARIES ${OVR_LIBRARY})

find_package(Threads REQUIRED)
set(OVR_LIBRARIES ${OVR_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
if(APPLE)
    find_package(OpenGL REQUIRED)
    find_library(COCOA_LIBRARY NAMES Cocoa)
    find_library(IOKIT_LIBRARY NAMES IOKit)
    set(OVR_LIBRARIES ${OVR_LIBRARIES} ${OPENGL_LIBRARIES} ${COCOA_LIBRARY} ${IOKIT_LIBRARY})
elseif(LINUX)
    find_package(X11 REQUIRED)
    find_package(Xinerama REQUIRED)
    find_package(UDev REQUIRED)
    set(OVR_INCLUDE_DIRS ${OVR_INCLUDES} ${XINERAMA_INCLUDE_DIR} ${X11_INCLUDE_DIR})
    set(OVR_LIBRARIES ${OVR_LIBRARIES} ${XINERAMA_LIBRARIES} ${UDEV_LIBRARIES} ${X11_LIBRARIES})
endif()

find_package_handle_standard_args(OVR REQUIRED_VARS OVR_INCLUDE_DIR OVR_LIBRARY)

mark_as_advanced(OVR_INCLUDE_DIR OVR_LIBRARY OVR_MSVC)
