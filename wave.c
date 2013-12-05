#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "timer.h"

#define MAX_SOURCE_SIZE (0x100000)

/*
float hcoef[] = ;
float gcoef[] = ;
*/
int filas(uchar *data, uchar *data2,
            uint m, uint n, uint channels,
            float *h, float *g, int fs,
            uint c, uint f);
int columnas(uchar *data, uchar *data2,
            uint m, uint n, uint channels,
            float *h, float *g, int fs,
            uint c, uint f);
// OpenCL kernel

const char* source =
	"__kernel \n"
	"void simpleMultiply(\n"
	"	__global float* outputC,\n"
	"	int widthA, int heightA,\n"
	"	int widthB, int heightB,\n"
	"	__global float* inputA,\n"
	"	__global float* inputB) {\n"
	"	   //Get global position in Y direction\n"
	"	   int row = get_global_id(1);\n"
	"	   //Get global position in X direction\n"
	"	   int col = get_global_id(0);\n"
	"	   float sum = 0.0f;\n"
	"	   //Calculate result of one element of Matrix C\n"
	"	   for (int i = 0; i < widthA; i++) {\n"
	"	      sum += inputA[row * widthA + i] * inputB[i * widthB + col];\n"
	"	   }\n"
	"	outputC[row*widthB+col] = sum;\n"
	"}\n";

int loadSource(const char *filename)
{
	FILE *fp;
	char *source_str;
	size_t source_size;

	/* Load the source code containing the kernel*/
	fp = fopen(filename, "r");
	if(!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
}


int debug(char *s, int error)
{
	if(error) {
		fprintf(stderr, "error en %s: %d\n",s,error);
		exit(1);
	}
}

void showDevice(cl_device_id device)
{
	char ret [1024];
	cl_uint ret_st;
	cl_ulong ret_ul;

	clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, ret, NULL);
	printf("CL_DEVICE_NAME: %s\n", ret);
	clGetDeviceInfo(device, CL_DEVICE_VENDOR, 1024, ret, NULL);
	printf("CL_DEVICE_VENDOR: %s\n", ret);
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ret_st), &ret_st, NULL);
	printf("CL_DEVICE_MAX_COMPUTE_UNITS: %d\n", ret_st);
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ret_ul), &ret_ul, NULL);
	printf("CL_DEVICE_LOCAL_MEM_SIZE: %lu\n", (unsigned long int)ret_ul);
	clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ret_ul), &ret_ul, NULL);
	printf("CL_DEVICE_GLOBAL_MEM_SIZE: %lu\n", (unsigned long int)ret_ul);
}

void opencl()
{
	cl_int ciErrNum;

	//Use the first platform
	cl_platform_id platform;
	ciErrNum = clGetPlatformIDs(1, &platform, NULL);
	debug("platform",ciErrNum);

	//Use the first device
	cl_device_id device;
	ciErrNum = clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&device,NULL);
	debug("device",ciErrNum);

	showDevice(device);

/*	cl_context_properties cps[3] =
		{CL_CONTEXT_PLATFORM,(cl_context_properties)platform, 0};

	// Create the context
	cl_context ctx = clCreateContext(cps,1,&device, NULL, NULL, &ciErrNum);
	debug("create context",ciErrNum);

	// Create the command queue
	cl_command_queue myqueue = clCreateCommandQueue(ctx, device, 0, &ciErrNum);
	debug("create queue",ciErrNum);


	// Allocate for A
	cl_mem bufferA = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
		A->w * A->h * sizeof(float), NULL, &ciErrNum);
	debug("alloc A",ciErrNum);

	// Copy A to device
	ciErrNum = clEnqueueWriteBuffer(myqueue, bufferA, CL_TRUE,
		0, A->w * A->h * sizeof(float), (void *) A->data, 0, NULL, NULL);
	debug("enqueue A",ciErrNum);


	// Allocate for B
	cl_mem bufferB = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
		B->w * B->h * sizeof(float), NULL, &ciErrNum);
	debug("alloc B",ciErrNum);


	// Copy B to device
	ciErrNum = clEnqueueWriteBuffer(myqueue, bufferB, CL_TRUE,
		0, B->w * B->h * sizeof(float), (void *) B->data, 0, NULL, NULL);
	debug("enqueue B",ciErrNum);

	// Allocate for C = A*B
	cl_mem bufferC = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
		A->h * B->w * sizeof(float), NULL, &ciErrNum);
	debug("alloc C",ciErrNum);


	// Read program from source ASCIIZ string
	cl_program myprog = clCreateProgramWithSource (
		ctx, 1, (const char **)&source, NULL, &ciErrNum);
	debug("read program",ciErrNum);


	// Compile program - pass NULL for device_list argument
	// so as to target all devices in the context
	ciErrNum = clBuildProgram(myprog, 0, NULL, NULL, NULL, NULL);
	debug("build program",ciErrNum);


	// Create the kernel
	cl_kernel mykernel = clCreateKernel(myprog, "simpleMultiply", &ciErrNum);
	debug("create program",ciErrNum);

	// Set the kernel args
	clSetKernelArg(mykernel, 0, sizeof(cl_mem), (void *)&bufferC);
	clSetKernelArg(mykernel, 1, sizeof(cl_int), (void *)&A->w);
	clSetKernelArg(mykernel, 2, sizeof(cl_int), (void *)&A->h);
	clSetKernelArg(mykernel, 3, sizeof(cl_int), (void *)&B->w);
	clSetKernelArg(mykernel, 4, sizeof(cl_int), (void *)&B->h);
	clSetKernelArg(mykernel, 5, sizeof(cl_mem), (void *)&bufferA);
	clSetKernelArg(mykernel, 6, sizeof(cl_mem), (void *)&bufferB);

	// Set local and global wkgroup sizes
	// Matrix sides are to be divis by localws[.]
	size_t localws[2] = {2, 2};
	size_t globalws[2] = {C->w, C->h};

//----------------------------------------------------------------------------
	// Execute the kernel
	ciErrNum = clEnqueueNDRangeKernel(myqueue, mykernel, 2, NULL,
			globalws, localws, 0, NULL, NULL);
	debug("execute",ciErrNum);
//----------------------------------------------------------------------------


	// Read the output data back to host
	ciErrNum = clEnqueueReadBuffer(
		myqueue, bufferC, CL_TRUE, 0, C->w * C->h *sizeof(float),
		(void *)C->data, 0, NULL, NULL);
	debug("enqueue C",ciErrNum);


	// Free OpenCL resources
	clReleaseKernel(mykernel);
	clReleaseProgram(myprog);
	clReleaseCommandQueue(myqueue);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	clReleaseContext(ctx);
*/
	free(platform);
	free(device);
}


// DAUB4 Recipes = {.4829, .8365, .2241, -.1294} /
// ESTA SIN NORMALIZAR. SUM(DAUB4)=1.4141; DAUB4/SUM(DAUB4) = db2

float phihaar[] = {.5, .5};
float psihaar[] = {.5, -.5};
uint nhaar = 2;

float db2[] = { 0.3415,    0.5915,    0.1585,   -0.0915};
float db3[] = { 0.2352,    0.5706,    0.3252,   -0.0955,   -0.0604,    0.0249};
float db4[] = { 0.1629,    0.5055,    0.4461,   -0.0198,   -0.1323,    0.0218,    0.0233,   -0.0075};



int seqGetCoefs(float **phicoefs, float **psicoefs, uint *ncoefs)
{
	*phicoefs = phihaar;
	*psicoefs = psihaar;
	*ncoefs = nhaar;
}

/*
 * int extractChannel(int channel, IplImage *img, IplImage *img2, float *signal)
{
	uchar *data = (uchar *)img->imageData;
	uchar *data2 = (uchar *)img2->imageData;
	int height    = img->height;
	int width     = img->width;

	int i;
	for(i = 0; i < height * width; i++) {
		*signal = (float) *(data + channel);
		*data2 = *signal;
		s++;
		data += 3;
		data2 += 1;
	}
}
*/

int extractChannel(IplImage *img, int channel, float *signal)
{
	uchar *data = (uchar *)img->imageData;
	int i;

	int height    = img->height;
	int width     = img->width;

	for(i = 0; i < height * width; i++) {
		*signal = (float) *(data + channel);
		signal++;
		data += 3;
	}
}

int secuencial(IplImage *img, IplImage *img2)
{
	float *psicoefs, *phicoefs; uint ncoefs;
	uint w, h, c, k;

    int i;

	seqGetCoefs((float **)&phicoefs, (float **)&psicoefs, &ncoefs);

    for(i=0; i < ncoefs; i++)
        printf("phi[%d]=%.2f  psi[%d]=%.2f\n",i,phicoefs[i],i,psicoefs[i]);

	w = img->width;
	h = img->height;
	c = img->nChannels;

	k=0;
/*	while(w > 2) {
		procesar(signal + k + channel, w, phicoefs, psicoefs, ncoefs, signal2);
		w /= 2;
	}
	*/

	filas(img->imageData, img2->imageData, w, h, c, phicoefs, psicoefs, ncoefs,0, 0);
  	columnas(img2->imageData, img->imageData, w, h, c, phicoefs, psicoefs, ncoefs,0, 0);

}

/* data = datos originales
   m = ancho de la fila
   n = altura de la columna
   ch = canal 0,1,2
   phi = coeficientes de h pasabajos
   psi = coeficientes de g pasaaltos
   fs = tamaño del filtro h o g
   data2 = buffer destino
   c = columna inicial
   f = fila inicial

*/
int filas(uchar *data, uchar *data2,
            uint m, uint n, uint channels,
            float *h, float *g, int fs,
            uint c, uint f)
{
	uint i,j,k, i1, j1, j2, j3, ch;

    for(ch = 0; ch < 3; ch++)
        for(i = f; i < f + n; i++) {
            i1 = i * channels;
            for(j = c; j < c + m/2; j++) {
                j1 = j * channels;
                j2 = j1 + i1 * m;
                data2[j2 + ch] = 0;
                for(k = 0; k < fs; k++) {
                    j3 = 2 * j1 + i1 * m + k * channels;
                    data2[j2 + ch] += data[j3 + ch] * h[k];
                }
            } // OK
            for(j = c + m/2; j < c + m; j++) {
                j1 = j * channels;
                j2 = j1 + i1 * m;
                data2[j2 + ch] = 0;
                for(k = 0; k < fs; k++) {
                    j3 = 2 * j1 + i1 * m + k * channels;
                    data2[j2 + ch] += data[j3 + ch] * h[k];
                }
            }
        } // OK
}

int columnas(uchar *data, uchar *data2,
            uint m, uint n, uint channels,
            float *h, float *g, int fs,
            uint c, uint f)
{
	uint i,j,k, i1, j1, i2, i3, ch;

    for(ch = 0; ch < 3; ch++)
        for(j = c; j < c + m; j++) {
            j1 = j * channels;
            for(i = f; i < f + n/2; i++) {
                i1 = i * channels;
                i2 = j1 + i1 * m;
                data2[i2 + ch] = 0;
                for(k = 0; k < fs; k++) {
                    i3 = 2 * i1 * m + j1 + k * channels;
                    data2[i2 + ch] += data[i3 + ch] * h[k];
                }
            }
           for(i = f + n/2; i < f + n; i++) {
                i1 = i * channels;
                i2 = j1 + i1 * m;
                data2[i2 + ch] = 0;
                for(k = 0; k < fs; k++) {
                    i3 = 2 * i1 * m + j1 + k * channels;
                    data2[i2 + ch] += data[i3 + ch] * h[k];
                    data2[i2 + ch] = data[i2 + ch];
                }
            }
        }

}


// This code executes on the OpenCL host
int main(int argc, char *argv[]) {

	IplImage *img,*img2;
	double elapsed;
	uint height, width, step, channels;

	uchar *data;
	float *signal;
	int i;
	uint channel;

	if(argc != 2) {
		printf("wave <imagen>\n");
		exit(1);
	}

  	// load an image
	img=cvLoadImage(argv[1],1);
	if(!img){
		printf("Could not load image file: %s\n",argv[1]);
		exit(0);
  	}

	// get the image data
	height    = img->height;
	width     = img->width;
	step      = img->widthStep;
	channels  = img->nChannels;
	data      = (uchar *)img->imageData;
	printf("Processing a %dx%d image with %d channels - step = %d\n",height,width,channels, step);

	img2 = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);
	img2->imageData = (uchar *) malloc(sizeof(uchar)*img->height * img->width * channels);
	img2->widthStep = step;
	img2->nChannels = channels;
	for(i=0; i < height * width * channels; i++)
        img2->imageData[i] = img->imageData[i];
	cvShowImage("INICIAL",img2);
	//cvWaitKey(0);


	//imgG = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);
	//imgB = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);
	//imgA = cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);

	//signal = (float *) malloc(sizeof(float) * img->height * img->width);

	//channel = 0;
	//extractChannel(img, channel, signal);

	timerOn();
	secuencial(img,img2);
	elapsed = timerOff();
	printf("Secuencial %lf ms\n",elapsed);

	opencl();
	cvShowImage("FINAL",img);

/*	cvShowImage("RED", imgR);
	cvShowImage("GREEN", imgG);
	cvShowImage("BLUE", imgB);
	cvShowImage("ALPHA", imgA);
*/
	cvWaitKey(0);


	timerOn();
	//parGetCoefs();
	elapsed = timerOff();
	printf("Paralelo %lf ms\n",elapsed);

}


