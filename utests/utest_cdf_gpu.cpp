/*
	unit test used to check if our cdf is producing the same result on cpu and gpu
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include <catch2/catch_test_macros.hpp>

#include "../src/histeq.h"
#include <cstdint>
#include <math.h>
#include "../src/vector3.h"

TEST_CASE("histeq CDF matches between CPU and GPU", "[histeq][gpu]")
{
	srand(1);
	// define grid dimensions for testing
	const float clipLimit = 0.1;
	const int binSize = 10;
	const vector3<std::size_t> subVolSpacing = {20, 20, 20};
	const vector3<std::size_t> volSize = {600, 500, 400};
	const vector3<std::size_t> subVolSize(11, 11, 11);

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[volSize[0] * volSize[1] * volSize[2]];
	for(int iIdx = 0; iIdx < (volSize[0] * volSize[1] * volSize[2]); iIdx ++)
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
	histHandler.calculate_cdf_gpu();

	// backup the version of the CDF calculated on the GPU
	float* cdf_bk = new float[histHandler.get_ncdf()];
	for (int iElem = 0; iElem < histHandler.get_ncdf(); iElem++)
	{
		cdf_bk[iElem] = histHandler.get_cdf(iElem);
	}

	// histogram calculation on CPU
	histHandler.calculate_cdf();
	int countNotSame = 0;
	for (int iElem = 0; iElem < histHandler.get_ncdf(); iElem++)
	{
		const float deltaVal = fabsf(cdf_bk[iElem] - histHandler.get_cdf(iElem));
		if (deltaVal >= 1e-6)
			countNotSame++;
	}

	// compare if results are the same
	INFO(countNotSame << " of " << histHandler.get_ncdf() << " CDF elements differ between CPU and GPU");
	REQUIRE(countNotSame == 0);

	delete[] inputVol;
	delete[] cdf_bk;
}
