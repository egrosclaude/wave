rm wave wave.o
gcc -o wave wave.c timer.c -l OpenCL -l opencv_core -l opencv_highgui -l opencv_imgproc -lm
