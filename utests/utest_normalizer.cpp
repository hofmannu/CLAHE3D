/*
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 10.03.2022
	Description: unit test for array normalization
*/

#include "../src/normalizer.h"
#include <iostream>

int main()
{
	normalizer<float> norm;

	norm.set_maxVal(1.2f);
	if (norm.get_maxVal() != 1.2f)
	{
		printf("Could not define maximum value for normalizer\n");
		throw "InvalidValue";
	}

	norm.set_minVal(0.2f);
	if (norm.get_minVal() != 0.2f)
	{
		printf("Could not define minimum value for normalizer\n");
		throw "InvalidValue";
	}

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

	// check lower boundary first
	if (minValTest != norm.get_minVal())
	{
		printf("Min value in normalized array is wrong\n");
		throw "InvalidValue";
	}


	if (maxValTest != norm.get_maxVal())
	{
		printf("Max value in normalized array is wrong\n");
		throw "InvalidValue";
	}

	delete[] testArray;
	return 0;
}