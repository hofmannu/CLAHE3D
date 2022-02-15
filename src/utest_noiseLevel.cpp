/*
	what happends when all values are below noise level?
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include "histeq.h"
#include <iostream>
#include <cstdint>
#include <fstream>
#include <chrono>

#include "vector3.h"

using namespace std;

int main()
{

	// define grid dimensions for testing
	const vector3<int64_t> volSize(600, 500, 400);
	const vector3<int64_t> subVolSize(31, 31, 31);
	const vector3<int64_t> subVolSpacing(20, 20, 20);
	const float clipLimit = 2.0;
	const int64_t binSize = 250;

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[volSize.elementMult()];
	for(int64_t iIdx = 0; iIdx < volSize.elementMult(); iIdx ++)
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);
		// this should generate a random number between 0 and 1

	const int64_t iBin = rand() % binSize;

	histeq histHandler;
	histHandler.set_nBins(binSize);
	histHandler.set_noiseLevel(clipLimit);
	histHandler.set_volSize(volSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	histHandler.set_overwrite(0);
	
	// histogram calculation on GPU
	histHandler.calculate_cdf();
	const float testValCpu = histHandler.get_cdf(iBin, 10, 10, 10);
	histHandler.equalize();

	float* outputVolCpu = histHandler.get_ptrOutput();
	for (int64_t iElem = 0; iElem < (volSize.elementMult()); iElem++)
	{
		if (!(outputVolCpu[iElem] == 0))
		{
			printf("CPU: All elements need to be zero now! I saw a %.1f\n", outputVolCpu[iElem]);
			throw "InvalidValue";
		}
	}

	printf("Looking good on CPU here\n");

	#if USE_CUDA
	histHandler.calculate_cdf_gpu();
	const float testValGpu = histHandler.get_cdf(iBin, 10, 10, 10);
	if (testValGpu != testValCpu)
	{	
		printf("CPU value: %.1f, GPU value: %.1f\n", testValCpu, testValGpu);
		throw "InvalidValue";
	}
	histHandler.equalize_gpu();

	float* outputVolGpu = histHandler.get_ptrOutput();
	for (int64_t iElem = 0; iElem < (volSize.elementMult()); iElem++)
	{
		if (!(outputVolGpu[iElem] == 0))
		{
			printf("GPU: All elements need to be zero now! I saw a %.1f\n", outputVolGpu[iElem]);
			throw "InvalidValue";
		}
	}

	printf("Looking good on GPU here\n");
	#endif

	delete[] inputVol;
		
	return 0;

}