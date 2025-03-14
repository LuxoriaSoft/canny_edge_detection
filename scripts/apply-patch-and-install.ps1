# apply-patch-and-install.ps1

# Define the vcpkg installation path
$vcpkgPath = "C:/vcpkg"

# Check if vcpkg is installed
if (Test-Path $vcpkgPath) {
    Write-Host "vcpkg is already installed, applying patch..."

    # Navigate to the vcpkg directory
    cd $vcpkgPath

    # Define the patch content
    $patchContent = @"
--- ports/opencv4/portfile.cmake
+++ ports/opencv4/portfile.cmake
@@ -112,6 +112,8 @@
     if(VCPKG_BUILD_TYPE STREQUAL "release")
         set(CMAKE_CXX_FLAGS_RELEASE "/DNDEBUG ${CMAKE_CXX_FLAGS_RELEASE}")
     endif()

+    set(OPENCV_DISABLE_ASSERTS ON CACHE BOOL "Disable OpenCV asserts")
+    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DNDEBUG")
"@

    # Write the patch content to a file
    $patchFilePath = "C:/vcpkg/patch.patch"
    $patchContent | Out-File -FilePath $patchFilePath

    # Apply the patch using git
    git apply $patchFilePath
} else {
    Write-Host "vcpkg is not installed, skipping patch application."
}

# Get the architecture from the environment variable
$vcpkgArch = $env:VCPKG_ARCH

# Install the package with the specified architecture
Write-Host "Installing package for architecture: $vcpkgArch"
C:/vcpkg/vcpkg install --triplet "${vcpkgArch}-windows-static"
