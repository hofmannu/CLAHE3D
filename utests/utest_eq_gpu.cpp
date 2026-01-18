/*
	checks if the equilization function delivers the same result for CPU and GPU
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

TEST_CASE("CPU and GPU equalization comparison", "[histeq][gpu][equalization]")
{
	// define grid dimensions for testing
	const vector3<std::size_t> volSize = {600, 500, 400};
	// initialize some parameters
	const vector3<std::size_t> subVolSize = {31, 31, 31};
	const vector3<std::size_t> subVolSpacing = {20, 20, 20};

	// generate input volume matrix and assign random values between 0.0 and 1.0
	float* inputVol = new float[volSize.elementMult()];
	for(int iIdx = 0; iIdx < volSize.elementMult(); iIdx ++)
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);


	histeq histHandler;
	histHandler.set_nBins(250);
	histHandler.set_noiseLevel(0.1);
	histHandler.set_volSize(volSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	histHandler.set_overwrite(0);
	
	// quickly check if nElements works
	REQUIRE(histHandler.get_nElements() == volSize.elementMult());

	// caluclate cummulative distribution function
	histHandler.calculate_cdf();

	// histogram calculation on GPU
	histHandler.equalize_gpu();

	// backup the result which we got from CPU
	float * outputBk = new float[histHandler.get_nElements()];
	for (int iElem = 0; iElem < histHandler.get_nElements(); iElem++)
	{
		outputBk[iElem] = histHandler.get_outputValue(iElem);
	}
	
	histHandler.equalize();
	bool isSame = 1;
	int counterNotSame = 0;
	for (int iElem = 0; iElem < histHandler.get_nElements(); iElem++)
	{
		const float deltaVal = abs(histHandler.get_outputValue(iElem) - outputBk[iElem]);
		if (deltaVal > 1e-6) // some inaccuracies might occur
		{
			isSame = 0;
			counterNotSame++;
			INFO("Difference found: CPU = " << outputBk[iElem] << ", GPU = " << histHandler.get_outputValue(iElem) << ", delta = " << deltaVal * 1e9);
		}
	}

	// check if results are the same, if not: complain
	REQUIRE(isSame);

	delete[] inputVol;
	delete[] outputBk;
}
