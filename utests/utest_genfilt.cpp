/*
	utest_genfilt.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 20.02.2022

	a generic filter class which applies a convolution with a kernel to the dataset
	can be used as a basis for any type of filter
*/

#include <catch2/catch_test_macros.hpp>

#include "../src/genfilt.h"
#include "../src/vector3.h"

TEST_CASE("genfilt reports correct sizes and bounded convolution", "[genfilt]")
{
	genfilt myFilt;

	// define the kernel size
	myFilt.set_kernelSize({5, 7, 9});
	vector3<std::size_t> range = myFilt.get_range();
	REQUIRE(range.x == 2);
	REQUIRE(range.y == 3);
	REQUIRE(range.z == 4);

	REQUIRE(myFilt.get_nKernel() == (5 * 7 * 9));

	myFilt.set_dataSize({100, 110, 120});
	vector3<std::size_t> paddedSize = myFilt.get_paddedSize();
	REQUIRE(paddedSize.x == 104);
	REQUIRE(paddedSize.y == 116);
	REQUIRE(paddedSize.z == 128);

	// generate an input array of 1s
	float* inputData = new float[myFilt.get_nData()];
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		inputData[iElem] = 1;
	}

	// generate a kernel array of 1s
	float* kernel = new float[myFilt.get_nKernel()];
	for (std::size_t iElem = 0; iElem < myFilt.get_nKernel(); iElem++)
	{
		kernel[iElem] = 1;
	}

	myFilt.set_dataInput(inputData);
	myFilt.set_kernel(kernel);

	myFilt.conv();

	// in this case all values should be between 0 and nKernel
	float* outputMatrix = myFilt.get_pdataOutput();
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		const float currVal = outputMatrix[iElem];
		REQUIRE(currVal >= 0.0f);
		REQUIRE(currVal <= myFilt.get_nKernel());
	}

	delete[] inputData;
	delete[] kernel;
}
