/* 
	test how our class handles a binarization task
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

TEST_CASE("binarization handling", "[histeq][binarization]")
{

	// define grid dimensions for testing
	const vector3<std::size_t> volSize(600, 500, 400);
	const float clipLimit = 0.1;
	const std::size_t binSize = 250;
	const vector3<std::size_t> subVolSize(31, 31, 31);
	const vector3<std::size_t> subVolSpacing(20, 20, 20);

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[volSize.elementMult()];
	for(int iIdx = 0; iIdx < volSize.elementMult(); iIdx ++)
	{
		inputVol[iIdx] = ((float) (iIdx % 2)) * 99.0 + 1.0;
	}
		
	// now our whole matrix is either 100 or 1

	// initialize some parameters

	histeq histHandler;
	histHandler.set_nBins(binSize);
	histHandler.set_noiseLevel(clipLimit);
	histHandler.set_volSize(volSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	histHandler.set_overwrite(0);
	
	histHandler.calculate_cdf();

	// check the min value in a single example bin
	REQUIRE(histHandler.get_minValBin(0, 0, 0) == 1.0);

	// check maximum value in an example bin
	REQUIRE(histHandler.get_maxValBin(0, 0, 0) == 100.0);

	// check if CDF is valid
	// all bins until last one should have value 0.5 in CDF
	for (int iBin = 0; iBin < binSize; iBin++)
	{
		if (iBin < (binSize - 1))
		{
			REQUIRE(histHandler.get_cdf(iBin, 0, 0, 0) == 0.0f);
		}
		else
		{
			REQUIRE(histHandler.get_cdf(iBin, 0, 0, 0) == 1.0f);
		}
	}

	// check example ICDFs
	// value 1.0 should return bin 0
	// value 

	histHandler.equalize();

	// the output should now be all either 1s or 0s with an even distribution
	float* outputVolCpu = histHandler.get_ptrOutput();
	int counterZero = 0;
	int counterOne = 0;
	for (int iElem = 0; iElem < (volSize.elementMult()); iElem++)
	{
		if (outputVolCpu[iElem] == 0.0)
		{
			counterZero ++;
		}
		else if (outputVolCpu[iElem] == 1.0)
		{
			counterOne ++;
		}
		else
		{
			INFO("All elements 0 or 1, most recent: input = " << inputVol[iElem] << ", output = " << outputVolCpu[iElem]);
			REQUIRE(false);
		}
	}

	REQUIRE(counterZero == counterOne);
	INFO("nZeros: " << counterZero << ", nOnes: " << counterOne);

	delete[] inputVol;
}