/*
	Simple runtime test to check if any errors occur during execution (no results checked)
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include "../src/histeq.h"
#include <iostream>
#include <cstdint>
#include <fstream>
#include <chrono>
#include "../src/vector3.h"


using namespace std;

int main()
{
	// define testing parameters
	const vector3<std::size_t> volSize(400, 200, 300);
	const float clipLimit = 0.01;
	const std::size_t binSize = 20;
	const vector3<std::size_t> subVolSize(31, 31, 31);
	const vector3<std::size_t> subVolSpacing(20, 20, 20);
	
	srand(1);

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[volSize.x * volSize.y * volSize.z];
	for(std::size_t iIdx = 0; iIdx < (volSize.x * volSize.y * volSize.z); iIdx ++)
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);
		// this should generate a random number between 0 and 1

	histeq histHandler;
	histHandler.set_nBins(binSize);
	histHandler.set_noiseLevel(clipLimit);
	histHandler.set_volSize(volSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	
	// histogram calculation on GPU
	histHandler.calculate_cdf();

	printf("Printing example bins\n");
	float oldBinVal = 0;
	for (std::size_t iBin = 1; iBin < binSize; iBin++)
	{
		const float delta = histHandler.get_cdf(iBin) - oldBinVal;
		printf("iBin = %d, Value = %.1f, Delta = %.4f\n", (int) iBin, histHandler.get_cdf(iBin), delta);
		oldBinVal = histHandler.get_cdf(iBin);
	}

	// first bin must be zero and last bin must be one
	for (std::size_t iSub = 0; iSub < histHandler.get_nSubVols(); iSub++)
	{
		if (histHandler.get_cdf(0, iSub) != 0.0)
		{
			printf("All initial bins must have zero value\n");
			throw "Invalid result";
		}

		const float deltaEnd = abs(1.0 - histHandler.get_cdf(binSize - 1, iSub));
		if (deltaEnd > 1e-6)
		{
			printf("All end bins must have one value: deviation: %.6f\n", deltaEnd);
			throw "Invalid result";
		}
	}

	histHandler.equalize();

	printf("CDF calculation took %.2f ms\n", histHandler.get_tCdf());
	printf("EQ calculation took %.2f ms\n", histHandler.get_tEq());
	
	delete[] inputVol;
		
	return 0;

}