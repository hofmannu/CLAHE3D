/*
	a simple test validating the performance of our median filter
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 10.03.2022
*/

#include <catch2/catch.hpp>
#include "../src/medianfilt.h"
#include <vector>
#include <algorithm>

TEST_CASE("median filter operations", "[medianfilt]")
{
	srand(time(NULL));

	const std::size_t nKernel = 5;
	const std::size_t range = (nKernel - 1) / 2;
	const std::size_t nx = 100;
	const std::size_t ny = 110;
	const std::size_t nz = 120;

	medianfilt myFilt;
	
	// set the size of the kernel and check if everything went smooth
	myFilt.set_kernelSize({nKernel, nKernel, nKernel});
	REQUIRE(myFilt.get_nKernel() == (nKernel * nKernel * nKernel));

	// set the size of the dataset and check if it was successful
	myFilt.set_dataSize({nx, ny, nz});
	REQUIRE(myFilt.get_nData() == (nx * ny * nz));

	// generate a volume with random values
	float* inputData = new float[myFilt.get_nData()];
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		inputData[iElem] = ((float) rand()) / ((float) RAND_MAX);
	}
	myFilt.set_dataInput(inputData);

	const float backupVal = inputData[101];

	myFilt.run();
	INFO("Median filtering took " << (myFilt.get_tExec() * 1e-3f) << " sec to execute");

	// check if input matrix remained the same
	REQUIRE(backupVal == inputData[101]);

	float* outputMatrix = myFilt.get_pdataOutput();
	
	// check that no value here exceeds the boundaries
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		const float currVal = outputMatrix[iElem];
		REQUIRE(currVal >= 0.0f);
		REQUIRE(currVal <= 1.0f);
	}

	// validate median at a single position
	std::vector<float> tempArray;
	
	vector3<std::size_t> testPos = {40, 40, 40};
	for (std::size_t iz = testPos.z - range; iz <= (testPos.z + range); iz++) 
	{
		for (std::size_t iy = testPos.y - range; iy <= (testPos.y + range); iy++) 
		{
			for (std::size_t ix = testPos.x - range; ix <= (testPos.x + range); ix++) 
			{
				const std::size_t idxVol = ix + nx * (iy + ny * iz); // index of volume
				tempArray.push_back(inputData[idxVol]);
			}
		}
	}
	sort(tempArray.begin(), tempArray.end());
	const std::size_t medianIdx = (nKernel * nKernel * nKernel - 1) / 2;

	const float valueProc = outputMatrix[testPos.x + nx * (testPos.y + ny * testPos.z)];
	const float testVal = tempArray[medianIdx];

	REQUIRE(valueProc == testVal);

	delete[] inputData;
}