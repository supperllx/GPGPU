
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#include <stdio.h>
#include <iostream>

#include "utils.h"


using namespace std;

#define CenterSize 4

__constant__ pixel Centers[CenterSize];  //define constant memory for Centers cache
cudaError_t cudaState;

void randomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}

__global__ void add(int n, float* x, float* y) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		y[i] += x[i];
	}
}

__global__ void gpu_labeling(int picSize, pixel* pic) {
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	float min_dis = FLT_MAX;
	int label = 0;
	for (int i = 0; i < CenterSize; ++i) {
		float dist = sqrtf(abs(powf(pic[index].R - Centers[i].R, 2) + powf(pic[index].G - Centers[i].G, 2) + powf(pic[index].B - Centers[i].B, 2)));
		if (dist <= min_dis) {
			min_dis = dist;
			label = i;
		}
		__syncthreads();
	}
	pic[index].label = label;

}

__global__ void gpu_updateCenter(unsigned int picSize, pixel* pic, unsigned int* temp ) {
	__shared__ unsigned int cache[CenterSize][4];
	int tid = threadIdx.x;
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < CenterSize) {
		for (int i = 0; i < 4; ++i) {
			cache[tid][i] = 0;
		}
	}
	__syncthreads();

	atomicAdd(&cache[pic[index].label][1], pic[index].R);
	atomicAdd(&cache[pic[index].label][2], pic[index].G);
	atomicAdd(&cache[pic[index].label][3], pic[index].B);
	atomicAdd(&cache[pic[index].label][0], 1);
	__syncthreads();

	if (tid == 0) {
		for (int i = 0; i < CenterSize; ++i) {
			int baseIndex = 4 * i;
			atomicAdd(&temp[baseIndex + 1], cache[i][1]);
			atomicAdd(&temp[baseIndex + 2], cache[i][2]);
			atomicAdd(&temp[baseIndex + 3], cache[i][3]);
			atomicAdd(&temp[baseIndex], cache[i][0]);
		}
	}
	__syncthreads();
}

int main() {
	showGPUinfo();

	Mat Img = imread("D:/img.bmp");
	const unsigned int picSize = Img.rows * Img.cols;

	pixel* pic;
	cudaMallocManaged(&pic, picSize * sizeof(pixel));
	imgToarray(Img, pic);

	pixel* cts;
	cudaMallocManaged(&cts, CenterSize * sizeof(pixel));
	initCenter(picSize, CenterSize, pic, cts);

	cudaState = cudaMemcpyToSymbol(Centers, cts, CenterSize * sizeof(pixel));
	if (cudaState != cudaSuccess)	cout << "error in copyToSymbol: " << cudaState << endl;

	int blockSize = 1024;
	const int gridSize = (picSize + blockSize - 1) / blockSize;
	cout << "block size: " << blockSize << endl;

	unsigned int* temp;
	cudaMallocManaged(&temp, CenterSize * 4 * sizeof(unsigned int));
	for (int i = 0; i < CenterSize * 4; ++i) {
		temp[i] = 0;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cout << "CUDA process start. " << endl;
	cudaEventRecord(start, 0);

	for (int round = 0; round < 10; ++round) {
		gpu_labeling << <gridSize, blockSize >> > (picSize, pic);
		cudaDeviceSynchronize();

		gpu_updateCenter << <gridSize, blockSize >> > (picSize, pic, temp);
		cudaDeviceSynchronize();

		for (int i = 0; i < CenterSize; ++i) {
			cts[i].R = temp[i * 4 + 1] / temp[i * 4];
			cts[i].G = temp[i * 4 + 2] / temp[i * 4];
			cts[i].B = temp[i * 4 + 3] / temp[i * 4];
		}
		
		cudaMemcpyToSymbol(Centers, cts, CenterSize * sizeof(pixel));
		cout << "Round " << round + 1 << " finished. " << "\r";
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time_usage;
	cudaEventElapsedTime(&time_usage, start, stop);

	cout << endl;
	cout << "CUDA process finished. " << endl;
	cout << "Kernel time usage: " << time_usage << "ms" << endl;

	cudaState = cudaMemcpyFromSymbol(cts, Centers, CenterSize * sizeof(pixel));
	if (cudaState != cudaSuccess)	cout << "error in copyFromSymbol: " << cudaState << endl;
	arrayToimg(Img, pic, cts, CenterSize);

	cudaFree(pic);
	cudaFree(cts);
	cudaFree(temp);
	cout << "Done." << endl;
}
