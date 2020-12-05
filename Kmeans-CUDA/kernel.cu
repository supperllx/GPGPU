
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

__global__ void gpu_labeling(int picSize, pixel* p) {
	unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
	float min_dis = FLT_MAX;
	int label = 0;
	for (int i = 0; i < CenterSize; ++i) {
		float dist = sqrtf(abs(powf(p[index].R - Centers[i].R, 2) + powf(p[index].G - Centers[i].G, 2) + powf(p[index].B - Centers[i].B, 2)));
		if (dist <= min_dis) {
			min_dis = dist;
			label = i;
		}
		__syncthreads();
	}
	p[index].label = label;

}

//__global__ void gpu_updateCenter()

int main() {
	showGPUinfo();

	Mat Img = imread("D:/img.bmp");
	const unsigned int picSize = Img.rows * Img.cols;

	pixel* pic;
	cudaMallocManaged(&pic, picSize * sizeof(pixel));
	imgToarray(Img, pic);

	pixel* cts = initCenter(picSize, CenterSize, pic);
	cudaState = cudaMemcpyToSymbol(Centers, cts, CenterSize * sizeof(pixel));
	if (cudaState != cudaSuccess)	cout << "error in copyToSymbol: " << cudaState << endl;

	int blockSize = 1024;
	int gridSize = (picSize + blockSize - 1) / blockSize;
	cout << "block size: " << blockSize << endl;

	cout << "CUDA process start. " << endl;
	for (int i = 0; i < 10; ++i) {
		gpu_labeling << <gridSize, blockSize >> > (picSize, pic);
		cudaDeviceSynchronize();
		pixel* new_center = updateCenter(picSize, CenterSize, pic, Centers);
		cudaMemcpyToSymbol(Centers, new_center, CenterSize * sizeof(pixel));
		cout << "Round " << i + 1 << " finished. " << "\r";
	}
	cout << endl;
	cout << "CUDA process finished. " << endl;
	cudaState = cudaMemcpyFromSymbol(cts, Centers, CenterSize * sizeof(pixel));
	if (cudaState != cudaSuccess)	cout << "error in copyFromSymbol: " << cudaState << endl;
	arrayToimg(Img, pic, cts, CenterSize);

	cudaFree(pic);
	cudaFree(cts);
	cout << "Done." << endl;
}
