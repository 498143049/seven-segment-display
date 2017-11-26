cmake_minimum_required(VERSION 3.5)
project(digital_linux_t)

#c++11
set(CMAKE_CXX_STANDARD 11)


# set open_cv
set(OpenCV_DIR "F:\\lib_win\\opencv\\mingw\\install")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
FIND_PACKAGE(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_calib3d opencv_video opencv_features2d opencv_ml opencv_highgui opencv_objdetect  )
#set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs  opencv_ml )
link_directories(
        F:/lib_win/opencv/mingw/install/x64/mingw/bin
)

## set boost
set(BOOST_ROOT "F:\\lib_win\\boost_1_64_0")
FIND_PACKAGE(Boost COMPONENTS system filesystem regex REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

include_directories("include")

set(SOURCE_FILES_train main.cpp Include/jsoncpp.cpp Include/WatershedSegmenter.hpp  Include/LBP.CPP Include/DigtalLocate.cpp  Include/Histogram1D.cpp  Include/tool.cpp  )
set(SOURCE_FILES main2.cpp Include/jsoncpp.cpp  Include/WatershedSegmenter.hpp  Include/LBP.CPP Include/DigtalLocate.cpp  Include/Histogram1D.cpp Include/tool.cpp test.cpp test.h)

set(SOURCE_FILES1 main3.cpp Include/jsoncpp.cpp Include/WatershedSegmenter.hpp  Include/LBP.CPP Include/DigtalLocate.cpp  Include/Histogram1D.cpp  Include/tool.cpp )

add_subdirectory(test)

add_executable(digital_linux_t ${SOURCE_FILES})

add_executable(digital_linux_train ${SOURCE_FILES_train})
add_executable(digital_linux_test_correcange ${SOURCE_FILES1})






target_link_libraries(digital_linux_t ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(digital_linux_t ${Boost_LIBRARIES} )

TARGET_LINK_LIBRARIES(digital_linux_test_correcange ${Boost_LIBRARIES} )
target_link_libraries(digital_linux_test_correcange ${OpenCV_LIBS})

TARGET_LINK_LIBRARIES(digital_linux_train ${Boost_LIBRARIES} )
target_link_libraries(digital_linux_train ${OpenCV_LIBS})




