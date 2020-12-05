#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>

using namespace std;
using namespace cv;

struct pixel
{
	int R = 0;
	int G = 0;
	int B = 0;
	int label = -1;
	pixel() = default;
	pixel(int r, int g, int b) {
		this->R = r;
		this->G = g;
		this->B = b;
	}
	pixel(Vec3b color) {
		this->R = int(color[0]);
		this->G = int(color[1]);
		this->B = int(color[2]);
	}
	void setRGB(int r, int g, int b) {
		this->R = r;
		this->G = g;
		this->B = b;
	}
};

void showGPUinfo() {
	//show gpu info
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

	std::cout << "--------GPU INFO--------" << std::endl;
	std::cout << "Name: " << devProp.name << std::endl;
	std::cout << "Compute capability: " << devProp.major << "." << devProp.minor << std::endl;
	std::cout << "Shared memory per block: " << devProp.sharedMemPerBlock << " Bytes" << std::endl;
	std::cout << "Threads per warp: " << devProp.warpSize << std::endl;
	std::cout << "Max thread per mp: " << devProp.maxThreadsPerMultiProcessor << std::endl;
	//std::cout << "Max block per mp: " << devProp.maxBlocksPerMultiProcessor << std::endl;  // This is only for CUDA 11+ 
	std::cout << "Max thread per block: " << devProp.maxThreadsPerBlock << std::endl;
	std::cout << "Max grid dim: " << devProp.maxGridSize[0] << " " << devProp.maxGridSize[1] << " " << devProp.maxGridSize[2]
		<< std::endl;
	std::cout << "Max block dim: " << devProp.maxThreadsDim[0] << " " << devProp.maxThreadsDim[1] << " " << devProp.maxThreadsDim[2]
		<< std::endl;
	std::cout << "-------------------------" << std::endl;
}



pixel* initCenter(unsigned int picSize, int csize, pixel* pic) {
	default_random_engine e;
	uniform_int_distribution<unsigned int> dist(0, picSize);

	pixel* cts = new pixel[csize];
	for (int i = 0; i < csize; ++i) {
		unsigned int rand_index = dist(e);
		cts[i] = pic[rand_index];
	}

	return cts;
}

pixel* updateCenter(unsigned int picSzie, int csize, pixel* pic, pixel* centers) {
	vector<vector<unsigned int>> temp(csize, vector<unsigned int>(4, 0));
	for (unsigned int i = 0; i < picSzie; ++i) {
		int index = pic[i].label;
		temp[index][0] += 1;
		temp[index][1] += pic[i].R;
		temp[index][2] += pic[i].G;
		temp[index][3] += pic[i].B;
	}

	for (int i = 0; i < csize; ++i) {
		temp[i][1] /= temp[i][0];
		temp[i][2] /= temp[i][0];
		temp[i][3] /= temp[i][0];
	}
	pixel* new_centers = new pixel[csize];
	for (int i = 0; i < csize; ++i) {
		new_centers[i].R = temp[i][1];
		new_centers[i].G = temp[i][2];
		new_centers[i].B = temp[i][3];
	}

	return new_centers;
}

//bool isConverge(int csize, pixel* centers, vector<vector<unsigned int>> temp) {
//	for (int i = 0; i < csize; ++i) {
//	}
//}


void imgToarray(Mat img, pixel* pic) {
	//const unsigned int picSize = img.rows * img.cols;
	//pixel* pic = new pixel[picSize];

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			unsigned int index = i * img.cols + j;
			pic[index] = pixel(img.ptr<Vec3b>(i)[j]);
		}
	}
	cout << "Image loading to pixel array success. " << endl;
}

void arrayToimg(Mat img, pixel* pic, pixel* cts, int csize) {
	vector<vector<int>> colorMap = { {255,0,0}, {0,255,0}, {0,0,255}, {0, 255,255}, {255,255,0}, {255,255,255} };

	/*RNG rng(time(0));*/
	for (int i = 0; i < csize; ++i) {
		/*cts[i].R = rng.uniform(0, 255);
		cts[i].G = rng.uniform(0, 255);
		cts[i].B = rng.uniform(0, 255);*/
		cts[i].setRGB(colorMap[i][0], colorMap[i][1], colorMap[i][2]);
	}


	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			unsigned int index = i * img.cols + j;
			int label = pic[index].label;
			img.ptr<Vec3b>(i)[j][0] = cts[label].R;
			img.ptr<Vec3b>(i)[j][1] = cts[label].G;
			img.ptr<Vec3b>(i)[j][2] = cts[label].B;
		}
	}
	imwrite("D:/result.bmp", img);
	cout << "result output success. " << endl;
}

