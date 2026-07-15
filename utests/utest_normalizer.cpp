/*
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 10.03.2022
	Description: unit test for array normalization
*/

#include <catch2/catch_test_macros.hpp>

#include "../src/normalizer.h"
#include <cmath>

TEST_CASE("normalizer scales an array into [minVal, maxVal]", "[normalizer]")
{
	normalizer<float> norm;

	norm.set_maxVal(1.2f);
	REQUIRE(norm.get_maxVal() == 1.2f);

	norm.set_minVal(0.2f);
	REQUIRE(norm.get_minVal() == 0.2f);

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

	// the resulting array must hit both boundaries exactly
	REQUIRE(minValTest == norm.get_minVal());
	REQUIRE(maxValTest == norm.get_maxVal());

	delete[] testArray;
}

TEST_CASE("normalizer handles a constant array without producing NaN", "[normalizer]")
{
	// regression: a constant array has zero span, so 1/rangeArray was inf and
	// (value - min) * inf produced NaN for every element.
	normalizer<float> norm;
	norm.set_minVal(0.2f);
	norm.set_maxVal(1.2f);

	float data[4] = {3.0f, 3.0f, 3.0f, 3.0f};
	norm.normalize(data, 4);

	for (float v : data)
	{
		REQUIRE_FALSE(std::isnan(v));
		REQUIRE(v == norm.get_minVal()); // degenerate range maps everything to minVal
	}
}

TEST_CASE("normalizer handles a single-element array", "[normalizer]")
{
	// regression: max was seeded from array[1], an out-of-bounds read for a
	// single-element array.
	normalizer<float> norm;
	norm.set_minVal(0.0f);
	norm.set_maxVal(1.0f);

	float data[1] = {5.0f};
	norm.normalize(data, 1);

	REQUIRE_FALSE(std::isnan(data[0]));
	REQUIRE(data[0] == norm.get_minVal());
}
