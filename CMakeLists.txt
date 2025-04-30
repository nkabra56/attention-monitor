cmake_minimum_required(VERSION 3.10)
project(attention_demo)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tell CMake where vcpkg toolchain is (only if not already in your cache)
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
endif()

# Find OpenCV — add 'video' to the list so optical‐flow symbols are available
find_package(OpenCV REQUIRED
    COMPONENTS
      core
      imgproc
      highgui
      video       
      videoio
      dnn
      face
)

# Include headers
include_directories(${OpenCV_INCLUDE_DIRS} include)

# Our executable
add_executable(attention_demo
    src/main.cpp
    src/AttentionMonitor.cpp
)

# Link against all the required OpenCV libs
target_link_libraries(attention_demo
    PRIVATE ${OpenCV_LIBS}
)
