cmake_minimum_required(VERSION 3.23)
project(GrabCutEdgeDetection)

# Set C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Handle OpenCV on different platforms
if(DEFINED ENV{VCPKG_INSTALLED_DIR})
    # If vcpkg is used, point to the vcpkg OpenCV directory
    set(OpenCV_ROOT "$ENV{VCPKG_INSTALLED_DIR}/${CMAKE_VCPKG_TARGET_TRIPLET}/share/opencv4")
elseif(APPLE)
    # If on macOS, Homebrew might be used to install OpenCV
    find_package(OpenCV REQUIRED PATHS /usr/local/opt/opencv4 /opt/homebrew/opt/opencv4)
elseif(WIN32)
    # If on Windows, use OpenCV path from vcpkg or from a custom path
    find_package(OpenCV REQUIRED PATHS "C:/path/to/vcpkg/installed/x64-windows/share/opencv4")
else()
    # Default OpenCV search if the platform is neither macOS nor Windows
    find_package(OpenCV REQUIRED)
endif()

# Ensure that OpenCV is found
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV and make sure it's in the CMake search path.")
endif()

# Set MSVC runtime library (for Windows)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Add the executable
add_executable(grabcut_edge_detection grabcut_edge_detection.cpp)

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
