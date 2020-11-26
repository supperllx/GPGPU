
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <sys/time.h>

using namespace std;

#define matrix_size 1024*1024
long long CPU_time, GPU_time;
void GPU_big_dot(float *c, float *a, float *b, int N);

float CPU_big_dot(float* a, float* b, int N) {
    float res;
    for (int i = 0; i < N; ++i) {
        res += a[i] * b[i];
    }
    return res;
}

__global__ void dotKernel(float* c, float* a, float* b) {
    int t_id = blockIdx.x*blockDim.x + threadIdx.x;
    c[t_id] = a[t_id] * b[t_id];
}

void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

float getSum(float* data, int size) {
    float res = 0;
    for (int i = 0; i < size; ++i) {
        res += data[i];
    }
    return res;
}

long long start_timer(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000000 + tv.tv_usec;
}

long long stop_timer(long long start_time, string name){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
    // printf("%s: %.5f sec\n", name, ((float) (end_time - start_time)) / (1000 * 1000));
    cout<<name<<": "<< (float(end_time - start_time)) / (1000 * 1000) <<endl;
    return end_time - start_time;
}

int main()
{
    cudaError_t cudaStatus;

    //init test matrix
    float a[matrix_size];
    float b[matrix_size];
    float c[matrix_size];
    randomInit(a, matrix_size);
    randomInit(b, matrix_size);
    long long start_time, stop_time;
    //CPU process
    cout<<"CPU process start!"<<endl;
    start_time = start_timer();
    float cpu_res;
    cpu_res = CPU_big_dot(a, b, matrix_size);
    cout<<"CPU final res: "<<cpu_res<<endl;
    CPU_time = stop_timer(start_time, "CPU time usage");
    cout<<"CPU process finished"<<endl;

    cout << "-------------------------" << endl;
    //GPU process
    cout<<"GPU process start!"<<endl;
    start_time = start_timer();
    GPU_big_dot(c, a, b, matrix_size);
    GPU_time = stop_timer(start_time, "GPU time usage");
    float GPU_res = getSum(c, matrix_size);
    cout << "GPU final res: " << GPU_res << endl;
    // cout << "main function finished" << endl;
    cout << "-------------------------" << endl;
    if (abs(cpu_res - GPU_res) <= 1.0e-6)   cout<<"correctness checking pass!"<<endl;
    else cout<<"correctness checking failed!"<<endl;
    cout<<"Speedup: "<<float(CPU_time)/ GPU_time<<endl;
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

void GPU_big_dot(float* c, float* a, float* b, int N) {
    cudaError_t cudaStatus;
    long long start_time, stop_time;
    int devID = 0;
    cudaStatus = cudaSetDevice(devID);
    if (cudaStatus != cudaSuccess) {
        cout << "cudaSetDevice failed" << endl;
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp devProp;
    cudaStatus = cudaGetDeviceProperties(&devProp, devID);
    if (cudaStatus != cudaSuccess) {
        cout << "cudaGetDeviceProperties failed" << endl;
        exit(EXIT_FAILURE);
    }
    //show gpu info
    // cout << "--------GPU INFO--------" << endl;
    // cout << "Name: " << devProp.name << endl;
    // cout << "Compute capability: " << devProp.major << "." << devProp.minor << endl;
    // cout << "Shared memory per block: " << devProp.sharedMemPerBlock << endl;
    // cout << "Threads per warp: " << devProp.warpSize << endl;
    // cout << "Max block per mp: " << devProp.maxBlocksPerMultiProcessor << endl;
    // cout << "Max thread per block: " << devProp.maxThreadsPerBlock << endl;
    // cout << "Max grid dim: " << devProp.maxGridSize[0] << " " << devProp.maxGridSize[1] << " " << devProp.maxGridSize[2]
    //     << endl;
    // cout << "Max block dim: " << devProp.maxThreadsDim[0] << " " << devProp.maxThreadsDim[1] << " " << devProp.maxThreadsDim[2]
    //     << endl;
    // cout << "-------------------------" << endl;

    int block_size = (devProp.major < 2) ? 512 : 1024;

    size_t size = N * sizeof(float);
    //allocate dev memory
    start_time = start_timer();
    float* dev_a, * dev_b, * dev_c;

    cudaStatus = cudaMalloc((void**)&dev_a, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(EXIT_FAILURE);
    }
    cudaStatus = cudaMalloc((void**)&dev_b, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(EXIT_FAILURE);
    }
    cudaStatus = cudaMalloc((void**)&dev_c, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(EXIT_FAILURE);
    }
    //copy memory from host(a, b) to devices(dev_a, dev_b)
    cudaStatus = cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cout << "cudaMemcpy failed!" << endl;
        exit(EXIT_FAILURE);
    }
    cudaStatus = cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cout << "cudaMemcpy failed!" << endl;
        exit(EXIT_FAILURE);
    }
    stop_time = stop_timer(start_time, "Device allocation and transfer time usage");
    block_size = (block_size <= N) ? block_size : N;

    dim3 dimBlock(block_size);
    // cout << "dimgrid: " << N / block_size << endl;
    dim3 dimGrid((N + block_size -1)/block_size);

    //kernel execution
    start_time = start_timer();
    dotKernel << <dimGrid, dimBlock >> > (dev_c, dev_a, dev_b);
    cudaDeviceSynchronize();
    stop_time = stop_timer(start_time, "kernel execution time usage");

    //copy memory back from devices(dev_c) to host(c)
    start_time = start_timer();
    cudaStatus = cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cout << "error in copy memory back!" << endl;
        exit(EXIT_FAILURE);
    }
    stop_time = stop_timer(start_time, "Data copy back time usage");
    cout << "finish kernel function in all threads" << endl;
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // cout << "done" << endl;
}
