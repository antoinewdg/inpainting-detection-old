cmake_minimum_required(VERSION 2.8)
project(copy_move_detector)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

set(SOURCE_FILES src/main.cpp
        src/utils.h
        src/shuffle.h
        src/patch_matcher.h
        )


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(copy_move_detector ${SOURCE_FILES})


# EXTERNAL LIBRAIRIES =====================

# OpenCV
FIND_PACKAGE(OpenCV 3.1 REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(copy_move_detector ${OpenCV_LIBS})