#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "timer.h"

#define MAX_SOURCE_SIZE (0x100000)



int filas(float *data, float *data2, uint m, uint n, uint channels,
            float *h, float *g, int fs, uint wh, uint ht);
int columnas(float *data, float *data2, uint m, uint n, uint channels,
            float *h, float *g, int fs, uint wh, uint ht);


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

struct wavelet {
#define MAXCOEFS 6
		float phi[MAXCOEFS];
		float psi[MAXCOEFS];
		uint ncoefs;
};

struct wavelet haar = {
	{.5f, .5f},
	{.7071067811865475f, -.7071067811865475f},
	2
};
/*
struct wavelet daub4 = {
	{ 0.4829f,    0.8365f, 	0.2241f,   -0.1294f },
	{ -.1294f, -.2241f, .8365f,  -.4829f },
	4
};	
*/
struct wavelet daub4 = {
	{ 0.3415f,    0.5915f, 	0.1585f,   -0.0915f },
	{ -0.0915f,   -0.1585f,  0.5915f,  -0.3415f },
	4
};	

struct wavelet db3 = {
	{ 0.2352,    0.5706,    0.3252,   -0.0955,   -0.0604,    0.0249},
	{ 0.2352,    0.5706,    0.3252,   -0.0955,   -0.0604,    0.0249},
	6
};

struct wavelet *wavelet;




/*
float phihaar[] = {.5f, .5f};
float psihaar[] = {.7071067811865475f, -.7071067811865475f};
uint nhaar = 2;

// DAUB4 Recipes = {.4829, .8365, .2241, -.1294} /
// ESTA SIN NORMALIZAR. SUM(DAUB4)=1.4141; DAUB4/SUM(DAUB4) = db2
// c0  c1 c2  c3 0 0 0 0 ....
// c3 -c2 c1 -c0 0 0 0 0 ....

float db2[] = { 0.3415,    0.5915,    0.1585,   -0.0915 };
float db3[] = { 0.2352,    0.5706,    0.3252,   -0.0955,   -0.0604,    0.0249};
float db4[] = { 0.1629,    0.5055,    0.4461,   -0.0198,   -0.1323,    0.0218,    0.0233,   -0.0075};
*/

/*
float phidb2[] = { 0.3415,    0.5915,    0.1585,   -0.0915 };
float psidb2[] = {-0.0915,    -0.1585,   0.5915,  -0.3415 };
uint ndb2 = 3;
*/

/*
float phidb2[] = { 0.3415,    0.5915, 	0.1585,   -0.0915};
float psidb2[] = { -.12940196, -.22415531,.83651652,  -.48295924};
uint ndb2 = 4;
*/
/* Stollnitz p.88
 * p = 1/4V2 (1+V3, 3+V3, 3-V3, 1-V3)
 * q = 1/4V2 (1-V3, -3+V3, 3+V3, -1-V3)
 * 15.45481364
26.76852247
7.17260403
-4.14110480
 * 
 * -4.14110480
-7.17260403
26.76852247
-15.45481364
---------- MAL! --------------
float phidb2[] = { 15.45481364, 26.76852247, 7.17260403, -4.14110480};
float psidb2[] = { -4.14110480, -7.17260403, 26.76852247, -15.45481364};
uint ndb2 = 4;
*/  
/*
int seqGetCoefs(float **phicoefs, float **psicoefs, uint *ncoefs)
{
	*phicoefs = phihaar;
	*psicoefs = psihaar;
	*ncoefs = nhaar;
}
*/


int seqGetCoefs(struct wavelet *w, float **phicoefs, float **psicoefs, uint *ncoefs)
{
	*phicoefs = w->phi;
	*psicoefs = w->psi;
	*ncoefs = w->ncoefs;
}

int img2signal(IplImage *img, float *signal)
{
	int i;
	for(i=0; i < img->width * img->height * img->nChannels; i++)
		signal[i] = (uchar) img->imageData[i];
}

int signal2img(float *signal, IplImage *img)
{
	int i;
	for(i=0; i < img->width * img->height * img->nChannels; i++)
		img->imageData[i] = signal[i];
}


int view(float *sig, IplImage *img)
{
	signal2img(sig,img);
	cvShowImage("Resultado",img);
	cvSaveImage("resultado.jpg",img,0);
	cvWaitKey(0);		
}

int zero(IplImage *img, float *sig)
{
	memset((void *) sig,0,img->width * img->height * sizeof(float) * img->nChannels);
}

int secuencial(struct wavelet *wave, IplImage *img1, IplImage *img2)
{
	float *psicoefs, *phicoefs; uint ncoefs;
	uint w, h, c, k;
	float *signal1, *signal2, *sigh;

    int i;

	seqGetCoefs(wave, (float **)&phicoefs, (float **)&psicoefs, &ncoefs);

    for(i=0; i < ncoefs; i++)
        printf("phi[%d]=%.2f  psi[%d]=%.2f\n",i,phicoefs[i],i,psicoefs[i]);

	w = img1->width;
	h = img1->height;
	c = img1->nChannels;
	
	signal1 = (float *) malloc(sizeof(float) * w * h * c);
	signal2 = (float *) malloc(sizeof(float) * w * h * c);
	
	img2signal(img1, signal1);
	cvShowImage("ORIGINAL",img1);
	cvWaitKey(0);	

	for(k=0; k<6; k++) {

		filas(signal1, signal2, w, h, c, phicoefs, psicoefs, ncoefs,
		img1->width,img1->height);
		columnas(signal2, signal1, w, h, c, phicoefs, psicoefs, ncoefs,
		img1->width,img1->height);

		view(signal1, img1);
		w /= 2;
		h /= 2;
	}

}

int filas(float *data, float *data2, uint m, uint n, uint channels,
            float *h, float *g, int fs, uint wh, uint ht)
{
	uint i,j,k, i1, j1, j2, j3, ch;

    for(ch = 0; ch < channels; ch++)
        for(i = 0; i < n; i++) {
  
            for(j = 0; j < m/2; j++) {
                j2 = (j + i * wh) * channels + ch;
                //assert(0 <= j2 && j2 <= 512 * 512 * 3);
                data2[j2] = 0.;
                for(k = 0; k < fs; k++) {
                    j3 = (2 * j + i * wh + k) * channels + ch;                    
                    //assert(0 <= j3);
                    //assert(j3 <= 512 * 512 * 3);
                    //printf("data2[j2]=%f, data[j3]=%f, h[k]=%f\n",data2[j2], data[j3], h[k]);
                    data2[j2] += data[j3] * h[k];
                    //assert(data2[j2] >= 0);
				}
            } // OK
            for(j = m/2; j < m; j++) {
                j2 = (j + i * wh) * channels + ch;
                //assert(0 <= j2 && j2 <= 512 * 512 * 3);
                data2[j2] = 0.;
                for(k = 0; k < fs; k++) {
                    j3 = (2 * (j - m/2) + i * wh + k) * channels + ch;
                    //assert(0 <= j3);
                    //assert(j3 <= 512 * 512 * 3);
                    //printf("data2[j2]=%f, data[j3]=%f, g[k]=%f\n",data2[j2], data[j3], g[k]);
                    data2[j2] += data[j3] * g[k];
                    //assert(data2[j2] >= 0);
                }
            }
        } // OK
}

int columnas(float *data, float *data2, uint m, uint n, uint channels,
            float *h, float *g, int fs, uint wh, uint ht)
{
	uint i,j,k, i1, j1, i2, i3, ch;

    for(ch = 0; ch < channels; ch++)
        for(j = 0; j < m; j++) {
			
            for(i = 0; i < n/2; i++) {
                i2 = (j + i * wh) * channels + ch;
                //assert(0 <= i2 && i2 <= 512 * 512 * 3);
                data2[i2] = 0;
                for(k = 0; k < fs; k++) {
                    i3 = (2 * i * wh + k * wh + j) * channels + ch;
                    //assert(0 <= i3);
                    //assert(i3 <= 512 * 512 * 3);
                    data2[i2] += data[i3] * h[k];
                    //assert(data2[i2] >= 0);
                }
            }
           for(i = n/2; i < n; i++) {
                i2 = (j + i * wh) * channels + ch;
                //assert(0 <= i2 && i2 <= 512 * 512 * 3);
                data2[i2] = 0;
                for(k = 0; k < fs; k++) {
                    i3 = (j + 2 * (i - n/2) * wh + k * wh) * channels + ch; 
                    //assert(0 <= i3);
                    //assert(i3 <= 512 * 512 * 3);
                    data2[i2] += data[i3] * g[k];
                    //assert(data2[i2] >= 0);
                }
            }
        }

}


int refilas(float *data, float *data2, uint m, uint n, uint channels,
			float *rh, float *rg, int fs, uint wh, uint ht)
{
}


// Prueba para ver funciones de conversion
int backandforth(IplImage *img1, IplImage *img2)
{
	float *signal1, *signal2;
	uint w, h, c, k;

	w = img1->width;
	h = img1->height;
	c = img1->nChannels;	

	signal1 = (float *) malloc(sizeof(float) * w * h * c);
	signal2 = (float *) malloc(sizeof(float) * w * h * c);
	
	img2signal(img1,signal1);
	signal2img(signal1,img2);
	
	cvShowImage("FORTH",img2);
	cvWaitKey(0);
	
}

int main(int argc, char *argv[]) {

	IplImage *img1,*img2;
	double elapsed;
	uint height, width, step, channels;

	uchar *data;
	int i;
	uint channel;

	if(argc != 3) {
		printf("wave <modelo> <imagen>\n");
		exit(1);
	}

	// wavelet model
	wavelet = (struct wavelet *) NULL;
	if(! strcmp(argv[1],"haar"))
		wavelet = &haar;
	if(! strcmp(argv[1],"daub4")) 
		wavelet = &daub4;
	if(! strcmp(argv[1],"db3")) 
		wavelet = &db3;
	if(! wavelet)
		debug("No se conoce el modelo (haar, daub4)", 0);
	

  	// load an image
	img1=cvLoadImage(argv[2],1);
	if(!img1){
		printf("Could not load image file: %s\n",argv[1]);
		exit(0);
  	}

	// get the image data
	height    = img1->height;
	width     = img1->width;
	step      = img1->widthStep;
	channels  = img1->nChannels;
	data      = (uchar *)img1->imageData;
	printf("Processing a %dx%d image with %d channels - step = %d\n",height,width,channels, step);

	// prepare second image object
	img2 = cvCreateImage(cvSize(width, height),IPL_DEPTH_8U,1);
	img2->imageData = (uchar *) malloc(sizeof(uchar)* height * width * channels);
	img2->widthStep = step;
	img2->nChannels = channels;

	// copy img1 into img2
	//for(i=0; i < height * width * channels; i++)
    //   img2->imageData[i] = img1->imageData[i];
	
	timerOn();
	secuencial(wavelet, img1, img2);
//backandforth(img1,img2);
	elapsed = timerOff();
	printf("Secuencial %e ms\n",elapsed);

//	opencl();
	timerOn();
	elapsed = timerOff();
	printf("Paralelo %lf ms\n",elapsed);

}


