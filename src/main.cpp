// File: test.cpp
// Function independent from matlab libraries used to debug
// since MATLAB debugging is a pain inn the ass
//
// Author: Urs Hofmann
// Mail: hofmannu@biomed.ee.ethz.ch
// Date: 23.11.2019
// Version: 1.0

#include "histeq.h"
#include "interpGrid.h"
#include <iostream>
#include <cstdint>
#include <fstream>
#include <chrono>

using namespace std;

int main(){

	printf("Testing CLAHE3D code functionality\n");

	// define grid dimensions for testing
	const uint64_t nZ = 600;
 	const uint64_t nX = 500;
	const uint64_t nY = 400;

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[nX * nY * nZ];
	for(uint64_t iIdx = 0; iIdx < (nX * nY * nZ); iIdx ++)
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);
		// this should generate a random number between 0 and 1

	// initialize some parameters
	const float clipLimit = 0.1;
	const uint64_t binSize = 250;
	const uint64_t subVolSize[3] = {31, 31, 31};
	const uint64_t subVolSpacing[3] = {20, 20, 20};
	const uint64_t gridSize[3] = {nZ, nX, nY};

	histeq histHandler;
	histHandler.setNBins(binSize);
	histHandler.setNoiseLevel(clipLimit);
	histHandler.setVolSize(gridSize);
	histHandler.setSizeSubVols(subVolSize);
	histHandler.setSpacingSubVols(subVolSpacing);
	histHandler.setData(inputVol);
	
	// histogram calculation on GPU
	auto startGpu = chrono::high_resolution_clock::now();
	histHandler.calculate_gpu();
	auto stopGpu = chrono::high_resolution_clock::now();
	auto durationGpu = chrono::duration_cast<chrono::milliseconds>(stopGpu - startGpu);
	const float testValGpu = histHandler.get_cdf(100, 5, 5, 5);

	// histogram calculation of CPU
	auto startCpu = chrono::high_resolution_clock::now();
	histHandler.calculate();
	auto stopCpu = chrono::high_resolution_clock::now();
	auto durationCpu = chrono::duration_cast<chrono::milliseconds>(stopCpu - startCpu);
	const float testValCpu = histHandler.get_cdf(100, 5, 5, 5);

	// compare if results are the same
	if (testValGpu != testValCpu)
	{
		printf("CPU value: %f, GPU value: %f\n", testValCpu, testValGpu);
		throw "InvalidResult";
	}
	else
	{
		printf("GPU and CPU deliver the same result for CDF!\n");
		printf("Time GPU: %d ms, Time CPU: %d ms\n", (int) durationGpu.count(), (int) durationCpu.count());
	}

	// loop over full volume
	printf("[clahe3d] running histogram equilization for each voxel\n");

	startCpu = chrono::high_resolution_clock::now();
	histHandler.equalize();
	stopCpu = chrono::high_resolution_clock::now();
	durationCpu = chrono::duration_cast<chrono::milliseconds>(stopCpu - startCpu);
	printf("Equilization took %d ms on CPU\n", durationCpu.count());

	// histHandler.equalize_gpu();

	
	delete[] inputVol;
		
	return 0;

}
