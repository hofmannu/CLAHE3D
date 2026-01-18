/* 
	File: utest_medianfilt_gpu.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 31.03.2022

	Description: Runs a median filter over volume on the GPU

*/

#include <catch2/catch.hpp>
#include <iostream>
#include "../src/medianfilt.h"
#include <random>

TEST_CASE("GPU median filter operations", "[medianfilt][gpu]")
{
	srand(time(NULL));

	const std::size_t nKernel = 5;
	const std::size_t range = (nKernel - 1) / 2;

	const std::size_t nx = 500;
	const std::size_t ny = 510;
	const std::size_t nz = 520;

	const std::size_t nxPadded = nx + nKernel - 1;
	const std::size_t nyPadded = ny + nKernel - 1;
	// const std::size_t nzPadded = nz + nKernel - 1;

	vector3<std::size_t> testPos = {40, 41, 45};

	medianfilt myFilt;
	myFilt.set_kernelSize({nKernel, nKernel, nKernel});
	REQUIRE(myFilt.get_nKernel() == (5 * 5 * 5));

	myFilt.set_dataSize({nx, ny, nz});
	REQUIRE(myFilt.get_nData() == (nx * ny * nz));

	// generate a volume with random values
	float* inputData = new float[myFilt.get_nData()];
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		inputData[iElem] = ((float) rand()) / ((float) RAND_MAX);
	}
	myFilt.set_dataInput(inputData);

	// backup an element of the input array
	const float backup_inputData = inputData[testPos.x + nx * (testPos.y + ny * testPos.z)];

	myFilt.run_gpu();
	INFO("Median filtering took " << myFilt.get_tExec() << " ms to execute (GPU)");

	REQUIRE(backup_inputData == inputData[testPos.x + nx * (testPos.y + ny * testPos.z)]);

	float* outputMatrix = myFilt.get_pdataOutput();
	REQUIRE(outputMatrix != nullptr);
	
	// check that no value here exceeds the boundaries
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		const float currVal = outputMatrix[iElem];
		REQUIRE(currVal >= 0.0);
		REQUIRE(currVal <= 1.00001);
	}

	// validate median at a single position
	std::vector<float> tempArray;
	
	for (std::size_t iz = (testPos.z - range); iz <= (testPos.z + range); iz++) 
	{
		for (std::size_t iy = (testPos.y - range); iy <= (testPos.y + range); iy++) 
		{
			for (std::size_t ix = (testPos.x - range); ix <= (testPos.x + range); ix++) 
			{
				const std::size_t idxVol = ix + nx * (iy + ny * iz); // index of volume
				tempArray.push_back(inputData[idxVol]);
			}
		}
	}
	sort(tempArray.begin(), tempArray.end());
	const int medianIdx = (nKernel * nKernel * nKernel - 1) / 2;

	const float valueProc = outputMatrix[testPos.x + nx * (testPos.y + ny * testPos.z)];
	const float testVal = tempArray[medianIdx];

	REQUIRE(valueProc == testVal);

	INFO("GPU test passed");

	myFilt.run();
	INFO("Median filtering took " << myFilt.get_tExec() << " ms to execute (CPU)");

	delete[] inputData;
}