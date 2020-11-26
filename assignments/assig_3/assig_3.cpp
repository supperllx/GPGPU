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
#include <iostream>
#include <vector>
#include <string>
#include "utils.h"

using namespace std;

#define N 40
#define BLOCK_SIZE 8

// cl_float *inputMatrix1;
// cl_float *inputMatrix2;
// cl_float *results;
cl_uint width = N;

int data = 0;
cl_float *inputMatrix1 = (cl_float *)malloc(sizeof(cl_float) * width * width);
cl_float *inputMatrix2 = (cl_float *)malloc(sizeof(cl_float) * width * width);
cl_float *results = (cl_float *)malloc(sizeof(cl_float) * width * width);

cl_int err;
cl_uint numPlatforms;
cl_platform_id myplatform;
cl_device_id mydevice;
cl_uint numDevices;
cl_context mycontext;
cl_command_queue mycommandq;
cl_kernel mykernelfunc;
cl_program myprogram;
cl_event prof_event;
cl_ulong start_time, end_time;
cl_mem gpuv_in1, gpuv_in2, gpuv_out;

char* loadProgSource(const char* filename, const char* preamble, size_t *sz) {
  FILE* fptr = NULL;
  size_t szSource, szPreamble, howmany;
  char* sourceString;

  // Open the OpenCL source code file
  fptr = fopen(filename, "r");
  szPreamble = strlen(preamble);

  // Get the length of the source code
  fseek(fptr, 0, SEEK_END);
  szSource = ftell(fptr);
  fseek(fptr, 0, SEEK_SET);

  // Allocate a buffer for the source code string and read it in
  sourceString = (char *) calloc(szSource + szPreamble+1, sizeof(char));
  howmany = fread((sourceString) + szPreamble, szSource, 1, fptr);
  fclose(fptr);
  *sz = szSource + szPreamble;
  sourceString[szSource + szPreamble] = '\0';
  return sourceString;
}

int initCL(){
    size_t mycontxtsize, kernelsize;	// size_t is unsigned long (64 bits).
    char *kernelsource;

    err = clGetPlatformIDs(1, &myplatform, &numPlatforms);
    if(err != CL_SUCCESS || numPlatforms <= 0)
    {
        cerr << "Failed to find any OpenCL platforms." << endl;
        return -1;
    }
    else    cout << "Success getting OpenCL platform" << endl;

    // char buffer[10240];
    // clGetPlatformInfo(myplatform, CL_PLATFORM_PROFILE, 102400, buffer, NULL);
    // printf("PROFILE: %s\n", buffer);
    // clGetPlatformInfo(myplatform, CL_PLATFORM_VERSION, 102400, buffer, NULL);
    // printf("VERSION: %s\n", buffer);
    // clGetPlatformInfo(myplatform, CL_PLATFORM_NAME, 102400, buffer, NULL);
    // printf("NAME: %s\n", buffer);
    // clGetPlatformInfo(myplatform, CL_PLATFORM_VENDOR, 102400, buffer, NULL);
    // printf("VENDOR: %s\n", buffer);
    // clGetPlatformInfo(myplatform, CL_PLATFORM_EXTENSIONS, 102400, buffer, NULL);
    // printf("EXTENSIONS: %s\n", buffer);

    err = clGetDeviceIDs(myplatform, CL_DEVICE_TYPE_GPU, 1, &mydevice, &numDevices);
    if(err != CL_SUCCESS || numDevices <=0){
        cerr << "Failed to find any OpenCL devices" << endl;
        return -1;
    }
    else{
        char buffer[10240];
        cl_uint buf_uint;
        cl_ulong buf_ulong;
        clGetDeviceInfo(mydevice, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_NAME = %s\n", buffer);
        clGetDeviceInfo(mydevice, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_VENDOR = %s\n", buffer);
        clGetDeviceInfo(mydevice, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
        printf("  DEVICE_VERSION = %s\n", buffer);
        clGetDeviceInfo(mydevice, CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
        printf("  DRIVER_VERSION = %s\n", buffer);
        clGetDeviceInfo(mydevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
        printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
        clGetDeviceInfo(mydevice, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
        printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
        clGetDeviceInfo(mydevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
        printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
    }

    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, 
	(cl_context_properties)myplatform, 0};

    mycontext = clCreateContext(props,1,&mydevice,NULL,NULL,&err);
    mycommandq = clCreateCommandQueue(mycontext,mydevice,CL_QUEUE_PROFILING_ENABLE,&err);
    kernelsource = loadProgSource("outProduct.cl","", &kernelsize);
    myprogram = clCreateProgramWithSource(mycontext, 1, (const char **)&kernelsource, NULL, &err);
    if(err != CL_SUCCESS){
        cout << "error: " << err << endl;
    }
    err = clBuildProgram(myprogram, 0, NULL, NULL, NULL, NULL);
    if(err != CL_SUCCESS)
    {
        printf("Error building program\n");
        cout << "err: " << err << endl;
        return 1;
    }
    mykernelfunc = clCreateKernel(myprogram,"outProduct",NULL);
    cout << "Initial success, BLOCK_SIZE: " << BLOCK_SIZE << endl;
    return 0;
}

void buffers(){
    gpuv_in1 = clCreateBuffer(mycontext,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, N*N*sizeof(float),inputMatrix1,NULL);
    gpuv_in2 = clCreateBuffer(mycontext,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, N*N*sizeof(float),inputMatrix2,NULL);
    gpuv_out = clCreateBuffer(mycontext,CL_MEM_WRITE_ONLY,N*N*sizeof(float),NULL,NULL);
}

void zoom()
{
    const int size = N;
    clSetKernelArg(mykernelfunc,0,sizeof(int), &size);
    clSetKernelArg(mykernelfunc,1,sizeof(cl_mem),(void *)&gpuv_out);
    clSetKernelArg(mykernelfunc,2,sizeof(cl_mem),(void *)&gpuv_in1);
    clSetKernelArg(mykernelfunc,3,sizeof(cl_mem),(void *)&gpuv_in2);

    size_t global_dim[2] = {N, N};
    size_t local_dim[2] = {BLOCK_SIZE, BLOCK_SIZE};
    err = clWaitForEvents(1, &prof_event);
    size_t return_bytes;
    err = clEnqueueNDRangeKernel(mycommandq, mykernelfunc, 2, NULL, global_dim, local_dim, 0, NULL, &prof_event);
    clFinish(mycommandq);
    err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &start_time, &return_bytes);
    err = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);
    double run_time = (double)(end_time - start_time)/1000000.0f;
    cout << "run time: " << run_time << endl;

    clEnqueueReadBuffer(mycommandq, gpuv_out, CL_TRUE, 0, N * N * sizeof(float), results, 0, NULL, NULL);

    float * cpu_res = (float *)malloc(sizeof(float) * width * width);
    mat_mul_cpu(N, cpu_res, inputMatrix1, inputMatrix2);
    if(check_result(N, results, cpu_res))   cout << "Check pass!" << endl;
    else{
        cout << "Check Fail!" << endl;
        print_mat(N, inputMatrix1);
        cout << "------------------" << endl;
        print_mat(N, inputMatrix2);
        cout << "------------------" << endl;
        print_mat(N, results);
        cout << "------------------" << endl;
        print_mat(N, cpu_res);
    }
}

void cleanup(int signo)
{
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

int main(){
    for(int y = 0; y < width; ++y){
        for(int x = 0; x < width; ++x){
            inputMatrix1[y * width + x] = data;
            inputMatrix2[y * width + x] = data;
            results[y * width + x] = 1.0f;
            data++;
        }
    }
    signal(SIGUSR1,cleanup);
    initCL();
    buffers();
    zoom();
    cleanup(SIGUSR1);
    return 0;
}