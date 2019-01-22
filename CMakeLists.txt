cmake_minimum_required(VERSION 3.9)
project(head_pose)

set(CMAKE_CXX_STANDARD 14)



set(OpenCV_DIR /Users/sravanchennuri/opencv-3.4.1/build)
set(GCC_COVERAGE_COMPILE_FLAGS "-O3 -mavx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )


add_subdirectory(/Users/sravanchennuri/dlib/ dlib_build)


find_package( OpenCV REQUIRED )

add_executable(head_pose main.cpp)
target_link_libraries( head_pose ${OpenCV_LIBS} dlib::dlib)
