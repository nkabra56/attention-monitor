cmake_minimum_required(VERSION 3.10)
project(attention_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
endif()

find_package(OpenCV REQUIRED
    COMPONENTS core imgproc highgui video videoio dnn face
)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(attention_demo
    src/main.cpp
    src/AttentionMonitor.cpp
)

target_link_libraries(attention_demo
    PRIVATE ${OpenCV_LIBS}
)
