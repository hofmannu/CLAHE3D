/*
	what happends when all values are below noise level?
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include "histeq.cuh"
#include "interpGrid.h"
#include <iostream>
#include <cstdint>
#include <fstream>
#include <chrono>

using namespace std;

int main()
{

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
	const float clipLimit = 2.0;
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
	histHandler.setOverwrite(0);
	
	// histogram calculation on GPU
	histHandler.calculate();
	histHandler.equalize();

	float* outputVolCpu = histHandler.get_ptrOutput();
	for (uint64_t iElem = 0; iElem < (nX * nY * nZ); iElem++)
	{
		if (!(outputVolCpu[iElem] == 0))
		{
			printf("CPU: All elements need to be zero now!\n");
			throw "InvalidValue";
		}
	}

	printf("Looking good on CPU here\n");

	histHandler.calculate_gpu();
	histHandler.equalize_gpu();

	float* outputVolGpu = histHandler.get_ptrOutput();
	for (uint64_t iElem = 0; iElem < (nX * nY * nZ); iElem++)
	{
		if (!(outputVolGpu[iElem] == 0))
		{
			printf("GPU: All elements need to be zero now!\n");
			throw "InvalidValue";
		}
	}

	printf("Looking good on GPU here\n");

	delete[] inputVol;
		
	return 0;

}