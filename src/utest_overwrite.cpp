/*
	overwrite test: checks if the overwrite function is working properly
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include "histeq.h"
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
	float* inputVolBk = new float[nX * nY * nZ];
	for(uint64_t iIdx = 0; iIdx < (nX * nY * nZ); iIdx ++)
	{
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);
		inputVolBk[iIdx] = inputVol[iIdx];
	}
		// this should generate a random number between 0 and 1

	// initialize some parameters
	const float clipLimit = 0.1;
	const uint64_t binSize = 250;
	const uint64_t subVolSize[3] = {31, 31, 31};
	const uint64_t subVolSpacing[3] = {20, 20, 20};
	const uint64_t gridSize[3] = {nZ, nX, nY};

	histeq histHandler;
	histHandler.set_nBins(binSize);
	histHandler.set_noiseLevel(clipLimit);
	histHandler.set_volSize(gridSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	histHandler.set_overwrite(0);
	
	// histogram calculation on GPU
	histHandler.calculate_cdf();
	histHandler.equalize();

	// check if input volume remained the same
	for (uint64_t iElem = 0; iElem < (nX * nY * nZ); iElem++)
	{
		if (inputVol[iElem] != inputVolBk[iElem])
		{
			printf("The input volume changed! Not acceptable.\n");
			throw "InvalidBehaviour";
		}
	}

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
		printf("Overwrite flag seems to work as expected!\n");
	}

	delete[] inputVol;
	delete[] inputVolBk;
		
	return 0;

}
