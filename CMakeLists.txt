cmake_minimum_required(VERSION 3.20)
project(attention_monitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Adjust OpenCV_DIR or use a vcpkg toolchain file as needed
find_package(OpenCV 4.11 REQUIRED
             COMPONENTS core imgproc highgui videoio dnn face)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(attention_demo
    src/AttentionMonitor.cpp
    src/main.cpp)

target_link_libraries(attention_demo PRIVATE ${OpenCV_LIBS})
