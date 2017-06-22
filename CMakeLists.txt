cmake_minimum_required(VERSION 3.5)
project(digital_linux_t)

#c++14
set(CMAKE_CXX_STANDARD 11)


# set open_cv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# set
find_package(TinyDNN  REQUIRED)
include_directories(${TinyDNN_INCLUDE_DIRS})

find_package(cereal  REQUIRED)

## set boost
FIND_PACKAGE(Boost COMPONENTS system filesystem regex REQUIRED)

FIND_PACKAGE(websocketpp REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

set(CMAKE_BUILD_TYPE release CACHE STRING "set build type to debug")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOMOC ON)

set(CNN_NO_SERIALIZATION OFF)
set(CMAKE_PREFIX_PATH "~/soft/qt/5.8/gcc_64/")
find_package(Qt5Core  REQUIRED)






include_directories("include")


set(SOURCE_FILES main.cpp Include/WatershedSegmenter.hpp  Include/LBP.CPP Include/DigtalLocate.cpp  Include/Histogram1D.cpp  Include/tool.cpp Include/web_interface.cpp Include/output_server.cpp )

add_subdirectory(test)

add_executable(digital_linux_t ${SOURCE_FILES})

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "  -Wextra -Wpedantic -pthread ${CMAKE_CXX_FLAGS}")
    message(STATUS "optional: -Wall -Wextra -Wpedantic -Werror")
endif(CMAKE_COMPILER_IS_GNUCXX)

qt5_use_modules(digital_linux_t Core Widgets Gui  WebEngine WebEngineWidgets WebChannel WebSockets)
TARGET_LINK_LIBRARIES(digital_linux_t ${Boost_LIBRARIES} )
target_link_libraries(digital_linux_t ${OpenCV_LIBS})
target_link_libraries(digital_linux_t libjsoncpp.a)
#target_link_libraries(digital_linux_t ${TinyDNN_LIBRARIES} ${TinyDNN_LIBRARY_DIRS} )



