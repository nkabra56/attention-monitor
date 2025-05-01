cmake_minimum_required(VERSION 3.20)
project(attention_monitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── vcpkg integration ──
if(DEFINED ENV{VCPKG_ROOT})
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
endif()

find_package(OpenCV 4 REQUIRED CONFIG
             COMPONENTS core imgproc highgui videoio dnn face)

add_executable(attention_demo
    src/AttentionMonitor.cpp
    src/main.cpp
)

# make include/ visible
target_include_directories(attention_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(attention_demo PRIVATE
    opencv::core
    opencv::imgproc
    opencv::highgui
    opencv::videoio
    opencv::dnn
    opencv::face
)
