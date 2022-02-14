/*
	Simple runtime test to check if any errors occur during execution (no results checked)
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
	const float clipLimit = 0.01;
	const uint64_t binSize = 20;
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
	
	// histogram calculation on GPU
	histHandler.calculate_cdf();

	printf("Printing example bins\n");
	float oldBinVal = 0;
	float deltaBin[binSize];
	for (uint64_t iBin = 1; iBin < binSize; iBin++)
	{
		const float delta = histHandler.get_cdf(iBin) - oldBinVal;
		printf("iBin = %d, Value = %.1f, Delta = %.4f\n", (int) iBin, histHandler.get_cdf(iBin), delta);
		oldBinVal = histHandler.get_cdf(iBin);
	}

	// first bin must be zero and last bin must be one
	for (uint64_t iSub = 0; iSub < histHandler.get_nSubVols(); iSub++)
	{
		if (histHandler.get_cdf(0, iSub) != 0.0)
		{
			printf("All initial bins must have zero value\n");
			throw "Invalid result";
		}

		if (histHandler.get_cdf(binSize - 1, iSub) != 1.0)
		{
			printf("All end bins must have one value\n");
			throw "Invalid result";
		}
	}

	histHandler.equalize();
	
	delete[] inputVol;
		
	return 0;

}