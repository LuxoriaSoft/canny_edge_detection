# apply-patch-and-install.ps1

# Define the vcpkg installation path
$vcpkgPath = "C:/vcpkg"

# Check if vcpkg is installed
if (Test-Path $vcpkgPath) {
    Write-Host "vcpkg is already installed, applying patch..."

    # Navigate to the vcpkg directory
    cd $vcpkgPath

    # Apply patch to OpenCV
    git apply << 'EOF'
    --- ports/opencv4/portfile.cmake
    +++ ports/opencv4/portfile.cmake
    @@ -112,6 +112,8 @@
        if(VCPKG_BUILD_TYPE STREQUAL "release")
            set(CMAKE_CXX_FLAGS_RELEASE "/DNDEBUG ${CMAKE_CXX_FLAGS_RELEASE}")
        endif()

    +    set(OPENCV_DISABLE_ASSERTS ON CACHE BOOL "Disable OpenCV asserts")
    +    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DNDEBUG")
    EOF
} else {
    Write-Host "vcpkg is not installed, skipping patch application."
}

# Install the package with the specified architecture
$vcpkgArch = $env:VCPKG_ARCH
Write-Host "Installing package for architecture: $vcpkgArch"
C:/vcpkg/vcpkg install --triplet $vcpkgArch-windows-static
