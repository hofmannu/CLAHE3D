/*
	overwrite test: checks if the overwrite function is working properly
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include "histeq.cuh"
#include <iostream>
#include <cstdint>
#include <fstream>
#include <chrono>

using namespace std;

int main(){

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
	histHandler.setOverwrite(0);
	
	// histogram calculation on GPU
	histHandler.calculate();
	histHandler.equalize();

	const float testVal1 = histHandler.get_cdf(120, 15, 2, 5);

	histHandler.equalize();

	const float testVal2 = histHandler.get_cdf(120, 15, 2, 5);

	if (testVal1 != testVal2)
	{
		printf("Test values are not identical!\n");
		throw "InvalidResult";
	}
	else
	{
		printf("Seems to work as expected!\n");
	}

	delete[] inputVol;
		
	return 0;

}
