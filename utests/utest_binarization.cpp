/*
	test how our class handles a binarization task
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include <catch2/catch_test_macros.hpp>

#include "../src/histeq.h"
#include <cstdint>
#include "../src/vector3.h"

TEST_CASE("histeq binarizes a two-valued volume into an even 0/1 split", "[histeq]")
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
	REQUIRE(histHandler.get_minValBin(0, 0, 0) == 1.0f);

	// check maximum value in an example bin
	REQUIRE(histHandler.get_maxValBin(0, 0, 0) == 100.0f);

	// check if CDF is valid
	// all bins until last one should have value 0 in CDF, the last one 1
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

	histHandler.equalize();

	// the output should now be all either 1s or 0s with an even distribution
	float* outputVolCpu = histHandler.get_ptrOutput();
	int counterZero = 0;
	int counterOne = 0;
	for (int iElem = 0; iElem < (volSize.elementMult()); iElem++)
	{
		const float outVal = outputVolCpu[iElem];
		INFO("input = " << inputVol[iElem] << ", output = " << outVal);
		REQUIRE((outVal == 0.0f || outVal == 1.0f));
		if (outVal == 0.0f)
			counterZero++;
		else
			counterOne++;
	}

	REQUIRE(counterZero == counterOne);

	delete[] inputVol;
}
