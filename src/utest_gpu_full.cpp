/*
	unit test used to check if our cdf is producing the same result on cpu and gpu
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

	srand(1);

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
	const uint64_t binSize = 10;
	const uint64_t subVolSize[3] = {11, 11, 11};
	const uint64_t subVolSpacing[3] = {20, 20, 20};
	const uint64_t volSize[3] = {nZ, nX, nY};

	const uint64_t iBin = rand() % binSize;

	histeq histHandler;
	histHandler.set_nBins(binSize);
	histHandler.set_noiseLevel(clipLimit);
	histHandler.set_volSize(volSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	histHandler.set_overwrite(0);

	// histogram calculation on GPU
	histHandler.calculate_cdf_gpu();
	histHandler.equalize_gpu();
	
	// backup the version of the CDF calculated with
	float* output_bk = new float[histHandler.get_nElements()];
	for (uint64_t iElem = 0; iElem < histHandler.get_nElements(); iElem++)
	{
		output_bk[iElem] = histHandler.get_outputValue(iElem);
	}

	// histogram calculation of CPU
	histHandler.calculate_cdf();
	histHandler.equalize();

	bool isSame = 1;
	uint64_t countNotSame = 0;
	for (uint64_t iElem = 0; iElem < histHandler.get_nElements(); iElem++)
	{
		const float deltaVal = abs(output_bk[iElem] - histHandler.get_outputValue(iElem));
		if (deltaVal > 1e-6)
		{
			isSame = 0;
			countNotSame++;
		}
	}

	// compare if results are the same
	if (!isSame)
	{
		const float percOff = ((float) countNotSame / ((float) histHandler.get_nElements())) * 100.0;
		printf("CPU and GPU results differ for %.1f percent of the elements\n", percOff);
		throw "InvalidResult";
	}
	else
	{
		printf("GPU and CPU deliver the same result for CDF!\n");
	}
	
	delete[] inputVol;
	delete[] output_bk;
		
	return 0;

}
