cmake_minimum_required(VERSION 3.10)
project(SLAM_Project)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH "/usr/local/")  # Update this path if needed
set(Open3D_DIR "/usr/local/lib/cmake/Open3D")  # Adjust this if necessary


find_package(OpenCV REQUIRED)
find_package(g2o REQUIRED)
find_package(DBoW2 REQUIRED)
find_package(Open3D REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${g2o_INCLUDE_DIRS} ${DBoW2_INCLUDE_DIRS} ${Open3D_INCLUDE_DIRS})

add_executable(SLAM_Project src/main.cpp)

target_link_libraries(SLAM_Project ${OpenCV_LIBS} g2o_core g2o_stuff DBoW2 Open3D::Open3D)
