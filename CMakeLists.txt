cmake_minimum_required(VERSION 3.5)
project(digital_linux_t)

#c++11
set(CMAKE_CXX_STANDARD 11)


# set open_cv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


## set boost
set(BOOST_ROOT "E:\\lib_win\\boost_1_64_0")
FIND_PACKAGE(Boost COMPONENTS system filesystem regex REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

include_directories("include")

set(SOURCE_FILES_train main.cpp Include/jsoncpp.cpp Include/WatershedSegmenter.hpp  Include/LBP.CPP Include/DigtalLocate.cpp  Include/Histogram1D.cpp  Include/tool.cpp  )
set(SOURCE_FILES main2.cpp Include/jsoncpp.cpp  Include/WatershedSegmenter.hpp  Include/LBP.CPP Include/DigtalLocate.cpp  Include/Histogram1D.cpp Include/tool.cpp  )

set(SOURCE_FILES1 main3.cpp Include/jsoncpp.cpp Include/WatershedSegmenter.hpp  Include/LBP.CPP Include/DigtalLocate.cpp  Include/Histogram1D.cpp  Include/tool.cpp )

add_subdirectory(test)

add_executable(digital_linux_t ${SOURCE_FILES})
add_executable(digital_linux_test_correcange ${SOURCE_FILES1})
add_executable(digital_linux_train ${SOURCE_FILES_train})





target_link_libraries(digital_linux_t ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(digital_linux_t ${Boost_LIBRARIES} )

TARGET_LINK_LIBRARIES(digital_linux_test_correcange ${Boost_LIBRARIES} )
target_link_libraries(digital_linux_test_correcange ${OpenCV_LIBS})

TARGET_LINK_LIBRARIES(digital_linux_train ${Boost_LIBRARIES} )
target_link_libraries(digital_linux_train ${OpenCV_LIBS})




