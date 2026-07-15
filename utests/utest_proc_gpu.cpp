/*
	Simple runtime test to check if any errors occur during GPU execution
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include <catch2/catch_test_macros.hpp>

#include "../src/histeq.h"
#include <cstdint>
#include "../src/vector3.h"

TEST_CASE("histeq GPU pipeline runs without error", "[histeq][gpu]")
{
	// define grid dimensions for testing
	const vector3<std::size_t> volSize(600, 500, 400);
	const float clipLimit = 0.1;
	const int binSize = 20;
	const vector3<std::size_t> subVolSize(31, 31, 31);
	const vector3<std::size_t> subVolSpacing(20, 20, 20);

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[volSize.elementMult()];
	for(int iIdx = 0; iIdx < (volSize.elementMult()); iIdx ++)
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
	REQUIRE_NOTHROW(histHandler.calculate_cdf_gpu());
	REQUIRE_NOTHROW(histHandler.equalize_gpu());

	delete[] inputVol;
}
