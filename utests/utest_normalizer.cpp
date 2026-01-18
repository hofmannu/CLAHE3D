/*
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 10.03.2022
	Description: unit test for array normalization
*/

#include <catch2/catch.hpp>
#include "../src/normalizer.h"
#include <iostream>

TEST_CASE("normalizer operations", "[normalizer]")
{
	normalizer<float> norm;

	SECTION("max value setting")
	{
		norm.set_maxVal(1.2f);
		REQUIRE(norm.get_maxVal() == 1.2f);
	}

	SECTION("min value setting")
	{
		norm.set_minVal(0.2f);
		REQUIRE(norm.get_minVal() == 0.2f);
	}

	SECTION("array normalization")
	{
		norm.set_maxVal(1.2f);
		norm.set_minVal(0.2f);

		// generate array with random values
		uint64_t nElements = 110;
		float* testArray = new float[110];
		for (uint64_t iElement = 0; iElement < nElements; iElement++)
			testArray[iElement] = ((float) rand()) / ((float) RAND_MAX);

		// normalize entire array
		norm.normalize(testArray, nElements);

		// check if normalization was successful
		float maxValTest = testArray[0];
		float minValTest = testArray[0];
		for (uint64_t iElement = 0; iElement < nElements; iElement++)
		{
			if (testArray[iElement] > maxValTest)
				maxValTest = testArray[iElement];

			if (testArray[iElement] < minValTest)
				minValTest = testArray[iElement];
		}

		REQUIRE(minValTest == norm.get_minVal());
		REQUIRE(maxValTest == norm.get_maxVal());

		delete[] testArray;
	}
}