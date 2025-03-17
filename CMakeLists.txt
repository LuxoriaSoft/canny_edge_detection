cmake_minimum_required(VERSION 3.23)
project(GrabCutEdgeDetection)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable CLR support for C++/CLI
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clr")

# Set the build type to Release
set(CMAKE_BUILD_TYPE Release)

# Add the NDEBUG flag for Release builds
add_definitions(-DNDEBUG)

# Set the OpenCV static flag
set(OpenCV_STATIC ON)

# Handle OpenCV on different platforms
if(WIN32)
    set(OpenCV_DIR "D:/opencv-install")
    find_package(OpenCV REQUIRED PATHS "D:/opencv-install" NO_DEFAULT_PATH)
elseif(APPLE)
    # If on macOS, Homebrew might be used to install OpenCV
    find_package(OpenCV REQUIRED PATHS /usr/local/opt/opencv4 /opt/homebrew/opt/opencv4)
else()
    # Default OpenCV search if the platform is neither macOS nor Windows
    find_package(OpenCV REQUIRED)
endif()

# Ensure that OpenCV is found
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV and make sure it's in the CMake search path.")
endif()

# Set MSVC runtime library (for Windows) based on build type
if(MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    endif()
endif()

# Add a new source file that includes the header file and uses the GrabCutWrapper class
add_library(grabcut_edge_detection SHARED GrabCutDLLEntrypoint.cpp)

# Add the includes folder to the include directories
target_include_directories(grabcut_edge_detection PUBLIC ${CMAKE_SOURCE_DIR}/includes)

# Link OpenCV libraries
target_link_libraries(grabcut_edge_detection ${OpenCV_LIBS})

# Optional: Add include directories for OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Set the output directory for the built executable
set_target_properties(grabcut_edge_detection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Suppress library conflicts for MSVC
if(MSVC)
    target_link_options(grabcut_edge_detection PRIVATE "/NODEFAULTLIB:library")
endif()
