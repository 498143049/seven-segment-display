cmake_minimum_required(VERSION 3.5)
project(testfile)

set(CMAKE_CXX_STANDARD 14)

# set open_cv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

## set boost
FIND_PACKAGE(Boost COMPONENTS system filesystem  REQUIRED )
include_directories(${Boost_INCLUDE_DIR})



include_directories("../include")
link_directories(${Boost_LIBRARY_DIR})

set(SOURCE_FILES1 main.cpp)

add_executable(testfile ${SOURCE_FILES1})
