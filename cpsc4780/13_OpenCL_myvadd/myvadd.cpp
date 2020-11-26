//
// myvadd.cpp - Hello World for OpenCL 
//
// This just adds two vectors of length 5.  The real purpose is to explain
// the standard OpenCL calling syntax. 
//
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_platform.h>
#include <CL/opencl.h>
#include "RGU.h"

#define N 5 

float vector1[N] = {0.0f,1.0f,2.0f,3.0f,4.0f};
float vector2[N] = {5.0f,6.0f,7.0f,8.0f,9.0f};
float outv[N];
size_t work[1] = {N};

// OpenCL globals.
cl_platform_id myplatform;
cl_context mycontext;
cl_device_id *mydevice;
cl_command_queue mycommandq;
cl_kernel mykernelfunc;
cl_program myprogram;
cl_mem gpuv_in1, gpuv_in2, gpuv_out;

void initCL()
{
int err;
size_t mycontxtsize, kernelsize;	// size_t is unsigned long (64 bits).
char *kernelsource;
unsigned int gpudevcount;

// Determine OpenCL platform
err = RGUGetPlatformID(&myplatform);
// Get number of GPU devices available on this platform:
err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,0,NULL,&gpudevcount);

// Create and load the device list:
mydevice = new cl_device_id[gpudevcount];
err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,gpudevcount,mydevice,NULL);

for (int i=0; i<gpudevcount; i++) {
	char buffer[10240];
	cl_uint buf_uint;
	cl_ulong buf_ulong;
	clGetDeviceInfo(mydevice[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
	printf("  DEVICE_NAME = %s\n", buffer);
	clGetDeviceInfo(mydevice[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
	printf("  DEVICE_VENDOR = %s\n", buffer);
	clGetDeviceInfo(mydevice[i], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
	printf("  DEVICE_VERSION = %s\n", buffer);
	clGetDeviceInfo(mydevice[i], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
	printf("  DRIVER_VERSION = %s\n", buffer);
	clGetDeviceInfo(mydevice[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
	printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
	clGetDeviceInfo(mydevice[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
	printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
	clGetDeviceInfo(mydevice[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
	printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
}

cl_context_properties props[] = {CL_CONTEXT_PLATFORM, 
	(cl_context_properties)myplatform, 0};

// Create a compute context
mycontext = clCreateContext(props,1,&mydevice[0],NULL,NULL,&err);
// Create a command queue
mycommandq = clCreateCommandQueue(mycontext,mydevice[0],0,&err);

// Load kernel file, prepend static info, and return total kernel size.
kernelsource = RGULoadProgSource("myvadd.cl","", &kernelsize);
// arg0: file name of kernel to load
// arg1: preamble to prepend, e.g., .h file
// arg2: final length of code 

// Create program object and loads source strings into it.
myprogram = clCreateProgramWithSource(mycontext,1,
	(const char **)&kernelsource, NULL, NULL);
// arg1: number of string pointers in array of string pointers
// arg2: array of string pointers
// arg3: array of size_ts with lengths of strings; 
//	 NULL==(all strings null-terminated)
// arg4: error code return

// Compile and link for all devices in context.
clBuildProgram(myprogram,0,NULL,NULL,NULL,NULL);
// arg1: number of devices in device list
// arg2: device list ptr; NULL == (use all devices in context)
// arg3: compile options; 
// arg4: callback function; called when compilation done; if NULL, suspend until
// arg5: data to callback function

// Create kernel object.
mykernelfunc = clCreateKernel(myprogram,"myvadd",NULL);
// arg1: kernel function name
// arg2: error code
}

void buffers()
{
// Create buffer objects == allocate mem on the card.
gpuv_in1 = clCreateBuffer(mycontext,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
	N*sizeof(float),vector1,NULL);
// RO means RO from a kernel; CL_MEM_COPY_HOST_PTR: alloc device memory 
// and copy data referenced by the host pointer; 
// arg4: error code return

gpuv_in2 = clCreateBuffer(mycontext,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
	N*sizeof(float),vector2,NULL);

gpuv_out = clCreateBuffer(mycontext,CL_MEM_WRITE_ONLY,N*sizeof(float),NULL,NULL);
// WO: written but not read from a kernel
}

void cleanup(int signo)
{
int i;
// Release GPU-allocated resources.
clReleaseProgram(myprogram);
clReleaseContext(mycontext);
clReleaseKernel(mykernelfunc);
clReleaseCommandQueue(mycommandq);
clReleaseMemObject(gpuv_in1);
clReleaseMemObject(gpuv_in2);
clReleaseMemObject(gpuv_out);
exit(0);
}

void zoom()
{
int i;

// Set parameters to the kernel, "mykernelfunc".
clSetKernelArg(mykernelfunc,0,sizeof(cl_mem),(void *)&gpuv_out);
clSetKernelArg(mykernelfunc,1,sizeof(cl_mem),(void *)&gpuv_in1);
clSetKernelArg(mykernelfunc,2,sizeof(cl_mem),(void *)&gpuv_in2);
// arg1: which argument (L-to-R)
// arg2: size of argument; can use sizeof(type) for mem objects
// arg3: argument *

// Launch the kernel.
clEnqueueNDRangeKernel(mycommandq,mykernelfunc,1,NULL,work,NULL,0,NULL,NULL);
// arg2: work dimension (of the grid)
// arg3: must be NULL; will be global work id offsets, instead of (0,0,...0)
// arg4: array of work dimension values giving number of work items in each
//       dim that will exec the kernel
// arg5: local work size - array of work dimension values giving work group
//       size in each dim; NULL = (OpenCL decides on work group sizes; 
//       Danger Will Robinson! OpenCL will make *BAD* decisions on this!)
// arg6: number of events in event waitlist
// arg7: event waitlist ... commands that must complete before exec this one
// arg8: event ... returns event object that identifies this kernel execution
//       instance; event objects are unique

// Read back the results.
clEnqueueReadBuffer(mycommandq,gpuv_out,CL_TRUE,0,N*sizeof(float),outv,0,NULL,NULL); 
// arg1: buffer object
// arg2: blocking read
// arg3: offset
// arg4: size in bytes
// arg5: host ptr
// arg6: number of events in event waitlist
// arg7: event waitlist ... commands that must complete before exec this one
// arg8: event ... returns event object that identifies this kernel execution
//       instance; event objects are unique

for(i=0;i<N;i++) printf("%f ",vector1[i]);
printf("\n");
for(i=0;i<N;i++) printf("%f ",vector2[i]);
printf("\n");
for(i=0;i<N;i++) printf("%f ",outv[i]);
printf("\n");
}

int main(int argc, char** argv)
{
signal(SIGUSR1,cleanup);
initCL();
buffers();
zoom();
cleanup(SIGUSR1);
}
