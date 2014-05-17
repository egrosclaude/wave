cmake_minimum_required(VERSION 2.8)
project( wave )
find_package( OpenCL REQUIRED )

#SET(OPENCV_INCLUDE_DIR "/usr/include/opencv")
#SET(OPENCV_LIB_DIR "/usr/lib")
#SET(OPENCV_ROOT_DIR "/usr/include/opencv")
#SET(OpenCV_ROOT_DIR "/usr/include/opencv")
#find_package( OpenCV REQUIRED )

set(OPENCV_INCLUDE_DIRS "/usr/include/opencv")
set(OPENCV_LIBRARIES "/usr/lib")



include_directories( ${OPENCL_INCLUDE_DIR} ${OPENCV_INCLUDE_DIR})
add_executable( wave wave.c timer.c )
target_link_libraries( wave ${OpenCV_LIBS} ${OpenCL_LIBS} OpenCL m opencv_core opencv_highgui opencv_imgproc )
