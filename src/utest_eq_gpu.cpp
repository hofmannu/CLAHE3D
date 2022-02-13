/*
	checks if the equilization function delivers the same result for CPU and GPU
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
	histHandler.set_nBins(binSize);
	histHandler.set_noiseLevel(clipLimit);
	histHandler.set_volSize(gridSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	histHandler.set_overwrite(0);
	
	// quickly check if nElements works
	if (histHandler.get_nElements() != (nZ * nX * nY))
	{
		printf("Number of elements is incorrect\n");
		throw "InvalidValue";
	}

	// histogram calculation on GPU
	histHandler.calculate_cdf();
	histHandler.equalize_gpu();

	// backup the result which we got from CPI
	float * outputBk = new float[histHandler.get_nElements()];
	for (uint64_t iElem = 0; iElem < histHandler.get_nElements(); iElem++)
	{
		outputBk[iElem] = histHandler.get_outputValue(iElem);
	}
	
	histHandler.equalize();
	bool isSame = 1;
	uint64_t counterNotSame = 0;
	for (uint64_t iElem = 0; iElem < histHandler.get_nElements(); iElem++)
	{
		if (histHandler.get_outputValue(iElem) != outputBk[iElem])
		{
			bool isSame = 0;
			counterNotSame++;
		}
	}

	if (!isSame)
	{
		const float percOff = ((float) counterNotSame) / ((float) histHandler.get_nElements()) * 100.0;
		printf("Sir we had a few differences here for %.1f percent!\n", percOff);
		throw "InvalidValue";
	}
	else
	{
		printf("Everything went well, congratulations.\n");
	}

	delete[] inputVol;
	delete[] outputBk;
		
	return 0;

}
