
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define matrix_size 512*2048
#define dimsize 1024

using namespace std;

void GPU_big_dot_1(float* c, float* a, float* b, int N);
void GPU_big_dot_2(float* res, float* a, float* b, int N);

float recursiveReduce(float *data, int const size)
{
    if (size == 1) return data[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];

    return recursiveReduce(data, stride);
}

float CPU_big_dot(float* a, float* b, int N) {
    static float res;
    for (int i = 0; i < N; ++i) {
        res += a[i] * b[i];
    }
    return res;
}

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int t_id = threadIdx.x;
    c[t_id] = a[t_id] * b[t_id];
}

__global__ void dotKernel(float* c, float* a, float* b) {
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    c[t_id] = a[t_id] * b[t_id];
}

__global__ void dotKnenel_opt_1(float* c, float* a, float* b) {
    extern __shared__ float cache[];
    int local_id = threadIdx.x;
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    cache[local_id] = a[t_id] * b[t_id];
    __syncthreads();

    if (t_id >= matrix_size) return;
    /*for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((local_id % (2 * stride)) == 0)   cache[local_id] += cache[local_id + stride];
        __syncthreads();
    }*/
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * local_id;
        if (index < blockDim.x) cache[index] += cache[index + stride];
        __syncthreads();
    }
    if (local_id == 0)   c[blockIdx.x] = cache[0];

}

__global__ void dotKnenel_opt_2(float* res, float* a, float* b) {
    extern __shared__ float cache[];
    int local_id = threadIdx.x;
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    cache[local_id] = a[t_id] * b[t_id];
    __syncthreads();

    if (t_id >= matrix_size) return;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * local_id;
        if (index < blockDim.x) cache[index] += cache[index + stride];
        __syncthreads();
    }
    if (local_id == 0){
        atomicAdd(res, cache[0]);
    }
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

float CPU_Psum(float* data, int size) {
    float res = 0;
    for (int i = 0; i < size; ++i) res += data[i];
    return res;
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

    // // for(int i=0;i<matrix_size;++i)  cout<<a[i]<<", ";
    // // cout<<endl;
    // //special
    // cout<<"get_sum: "<<getSum(a, matrix_size)<<endl;
    // cout<<"special: "<< recursiveReduce(a, matrix_size)<<endl;;

    //CPU process
    cout << "------------------------" << endl;
    // cout << "CPU process start!" << endl;
    float CPU_res;
    CPU_res = CPU_big_dot(a, b, matrix_size);
    cout << "CPU final res: " << CPU_res << endl;
    // cout << "CPU process finished" << endl;

    cout << "------------------------" << endl;
    GPU_big_dot_1(c, a, b, matrix_size);
    float GPU_res_1 = CPU_Psum(c, matrix_size/dimsize + 1);
    std::cout << "GPU_res_1: " << GPU_res_1 << std::endl;

    float GPU_res_2 = 0;
    cout << "------------------------" << endl;
    GPU_big_dot_2(&GPU_res_2, a, b, matrix_size);
    cout << "GPU_res_2: " << GPU_res_2 << endl;

    cout << "------------------------" << endl;
    std::cout << "main function finished" << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

void GPU_big_dot_1(float* c, float* a, float* b, int N) {
    cudaError_t cudaStatus;
    int devID = 0;
    cudaStatus = cudaSetDevice(devID);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaSetDevice failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp devProp;
    cudaStatus = cudaGetDeviceProperties(&devProp, devID);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaGetDeviceProperties failed" << std::endl;
        exit(EXIT_FAILURE);
    }
    //show gpu info
    /*std::cout << "--------GPU INFO--------" << std::endl;
    std::cout << "Name: " << devProp.name << std::endl;
    std::cout << "Compute capability: " << devProp.major << "." << devProp.minor << std::endl;
    std::cout << "Shared memory per block: " << devProp.sharedMemPerBlock << std::endl;
    std::cout << "Threads per warp: " << devProp.warpSize << std::endl;
    std::cout << "Max block per mp: " << devProp.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Max thread per block: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max grid dim: " << devProp.maxGridSize[0] << " " << devProp.maxGridSize[1] << " " << devProp.maxGridSize[2]
        << std::endl;
    std::cout << "Max block dim: " << devProp.maxThreadsDim[0] << " " << devProp.maxThreadsDim[1] << " " << devProp.maxThreadsDim[2]
        << std::endl;
    std::cout << "-------------------------" << std::endl;*/

    int block_size = (devProp.major < 2) ? 512 : 1024;

    size_t size = N * sizeof(float);
    //allocate dev memory
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
        std::cout << "cudaMemcpy failed!" << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaStatus = cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    block_size = (block_size <= N) ? block_size : N;

    dim3 dimBlock(block_size);
    dim3 dimGrid(ceil(N / float(block_size)));
    // std::cout << "dimgrid: " << dimGrid.x << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    dotKnenel_opt_1 << <dimGrid, dimBlock, block_size*sizeof(float) >> > (dev_c, dev_a, dev_b);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float time_usage;
    cudaEventElapsedTime(&time_usage, start, stop);
    cout << "Kernel_1 time usage: " << time_usage << "ms"<< endl;

    //copy memory back from devices(dev_c) to host(c)

    cudaStatus = cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cout << "error in copy memory back!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // std::cout << "finish kernel function in all threads" << std::endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // std::cout << "done" << std::endl;
}

void GPU_big_dot_2(float* res, float* a, float* b, int N) {
    cudaError_t cudaStatus;
    int devID = 0;
    cudaStatus = cudaSetDevice(devID);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaSetDevice failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp devProp;
    cudaStatus = cudaGetDeviceProperties(&devProp, devID);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaGetDeviceProperties failed" << std::endl;
        exit(EXIT_FAILURE);
    }

    int block_size = (devProp.major < 2) ? 512 : 1024;

    size_t size = N * sizeof(float);
    //allocate dev memory
    float* dev_a, * dev_b;
    float* dev_res = 0;
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
    cudaStatus = cudaMalloc((void**)&dev_res, sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        exit(EXIT_FAILURE);
    }
    //copy memory from host(a, b) to devices(dev_a, dev_b)
    cudaStatus = cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed!" << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaStatus = cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    block_size = (block_size <= N) ? block_size : N;

    dim3 dimBlock(block_size);
    dim3 dimGrid(ceil(N / float(block_size)));
    // std::cout << "dimgrid: " << dimGrid.x << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    dotKnenel_opt_2 << <dimGrid, dimBlock, block_size * sizeof(float) >> > (dev_res, dev_a, dev_b);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float time_usage;
    cudaEventElapsedTime(&time_usage, start, stop);
    cout << "Kernel_2 time usage: " << time_usage << "ms"<< endl;
    // cout << "dev_res: " << *dev_res << endl;
    //copy memory back from devices(dev_res) to host(res)

    cudaStatus = cudaMemcpy(res, dev_res, sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cout << "error in copy memory back!" << std::endl;
        exit(EXIT_FAILURE);
    }
    // std::cout << "finish kernel function in all threads" << std::endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_res);

    // std::cout << "done" << std::endl;
}
