/*
	Simple runtime test to check if any errors occur during execution (no results checked)
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include <catch2/catch.hpp>
#include "../src/histeq.h"
#include <iostream>
#include <cstdint>
#include <fstream>
#include <chrono>
#include "../src/vector3.h"

using namespace std;

TEST_CASE("histogram equalization processing", "[histeq][proc]")
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

	INFO("Printing example bins");
	float oldBinVal = 0;
	for (std::size_t iBin = 1; iBin < binSize; iBin++)
	{
		const float delta = histHandler.get_cdf(iBin) - oldBinVal;
		INFO("iBin = " << iBin << ", Value = " << histHandler.get_cdf(iBin) << ", Delta = " << delta);
		oldBinVal = histHandler.get_cdf(iBin);
	}

	// first bin must be zero and last bin must be one
	for (std::size_t iSub = 0; iSub < histHandler.get_nSubVols(); iSub++)
	{
		REQUIRE(histHandler.get_cdf(0, iSub) == 0.0);

		const float deltaEnd = abs(1.0 - histHandler.get_cdf(binSize - 1, iSub));
		REQUIRE(deltaEnd < 1e-6);
	}

	histHandler.equalize();

	INFO("CDF calculation took " << histHandler.get_tCdf() << " ms");
	INFO("EQ calculation took " << histHandler.get_tEq() << " ms");
	
	delete[] inputVol;
}