/*
	full pipeline check: CDF + equalization must match between CPU and GPU
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include <catch2/catch_test_macros.hpp>

#include "../src/histeq.h"
#include <cstdint>
#include <chrono>
#include "../src/vector3.h"

TEST_CASE("histeq full CDF+equalization pipeline matches between CPU and GPU", "[histeq][gpu]")
{
	srand(1);

	// define grid dimensions for testing
	const vector3<std::size_t> volSize = {600, 500, 400};
	const float clipLimit = 0.1;
	const int binSize = 10;
	const vector3<std::size_t> subVolSize = {31, 31, 31};
	const vector3<std::size_t> subVolSpacing = {10, 10, 10};

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[volSize.elementMult()];
	for(int iIdx = 0; iIdx < volSize.elementMult(); iIdx ++)
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);
		// this should generate a random number between 0 and 1

	histeq histHandler;
	histHandler.set_nBins(binSize);
	histHandler.set_noiseLevel(clipLimit);
	histHandler.set_volSize(volSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	histHandler.set_overwrite(0);

	// full pipeline on GPU
	histHandler.calculate_cdf_gpu();
	histHandler.equalize_gpu();

	// backup the GPU output
	float* output_bk = new float[histHandler.get_nElements()];
	for (int iElem = 0; iElem < histHandler.get_nElements(); iElem++)
	{
		output_bk[iElem] = histHandler.get_outputValue(iElem);
	}

	// full pipeline on CPU
	histHandler.calculate_cdf();
	histHandler.equalize();

	int countNotSame = 0;
	for (int iElem = 0; iElem < histHandler.get_nElements(); iElem++)
	{
		const float deltaVal = abs(output_bk[iElem] - histHandler.get_outputValue(iElem));
		if (deltaVal > 1e-6)
			countNotSame++;
	}

	// compare if results are the same
	INFO(countNotSame << " of " << histHandler.get_nElements()
		<< " elements differ between CPU and GPU");
	REQUIRE(countNotSame == 0);

	delete[] inputVol;
	delete[] output_bk;
}
