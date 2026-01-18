/*
	overwrite test: checks if the overwrite function is working properly
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

using namespace std;

TEST_CASE("overwrite flag functionality", "[histeq][overwrite]")
{

	const vector3<std::size_t> volSize(600, 500, 300);
	const float clipLimit = 0.1;
	const std::size_t binSize = 250;
	const vector3<std::size_t> subVolSize(31, 31, 31);
	const vector3<std::size_t> subVolSpacing(20, 20, 20);
	

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[volSize.x * volSize.y * volSize.z];
	float* inputVolBk = new float[volSize.x * volSize.y * volSize.z];
	for(std::size_t iIdx = 0; iIdx < (volSize.x * volSize.y * volSize.z); iIdx ++)
	{
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);
		inputVolBk[iIdx] = inputVol[iIdx];
	}
		// this should generate a random number between 0 and 1

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
	histHandler.equalize();

	// check if input volume remained the same
	for (std::size_t iElem = 0; iElem < (volSize.x * volSize.y * volSize.z); iElem++)
	{
		REQUIRE(inputVol[iElem] == inputVolBk[iElem]);
	}

	const float testVal1 = histHandler.get_cdf(120, 15, 2, 5);

	histHandler.equalize();

	const float testVal2 = histHandler.get_cdf(120, 15, 2, 5);

	REQUIRE(testVal1 == testVal2);

	delete[] inputVol;
	delete[] inputVolBk;
}
