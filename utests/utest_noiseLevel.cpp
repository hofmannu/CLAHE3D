/*
	what happends when all values are below noise level?
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include <catch2/catch_test_macros.hpp>

#include "../src/histeq.h"
#include <cstdint>
#include "../src/vector3.h"

TEST_CASE("histeq zeroes the volume when everything is below noise level", "[histeq]")
{
	// define grid dimensions for testing
	const vector3<std::size_t> volSize(600, 500, 400);
	const vector3<std::size_t> subVolSize(31, 31, 31);
	const vector3<std::size_t> subVolSpacing(20, 20, 20);
	const float clipLimit = 2.0;
	const std::size_t binSize = 250;

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[volSize.elementMult()];
	for(std::size_t iIdx = 0; iIdx < volSize.elementMult(); iIdx ++)
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);
		// this should generate a random number between 0 and 1

	const std::size_t iBin = rand() % binSize;

	histeq histHandler;
	histHandler.set_nBins(binSize);
	histHandler.set_noiseLevel(clipLimit);
	histHandler.set_volSize(volSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	histHandler.set_overwrite(0);

	// histogram calculation on CPU
	histHandler.calculate_cdf();
	const float testValCpu = histHandler.get_cdf(iBin, 10, 10, 10);
	histHandler.equalize();

	float* outputVolCpu = histHandler.get_ptrOutput();
	for (std::size_t iElem = 0; iElem < (volSize.elementMult()); iElem++)
	{
		REQUIRE(outputVolCpu[iElem] == 0);
	}

	#if USE_CUDA
	histHandler.calculate_cdf_gpu();
	const float testValGpu = histHandler.get_cdf(iBin, 10, 10, 10);
	REQUIRE(testValGpu == testValCpu);
	histHandler.equalize_gpu();

	float* outputVolGpu = histHandler.get_ptrOutput();
	for (std::size_t iElem = 0; iElem < (volSize.elementMult()); iElem++)
	{
		REQUIRE(outputVolGpu[iElem] == 0);
	}
	#endif

	delete[] inputVol;
}
