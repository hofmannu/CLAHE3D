/*
	File: utest_genfilt_gpu.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 01.04.2022

	Description: runs the general filtering algorithm on the GPU and checks for correctness
*/

#include <catch2/catch.hpp>
#include "../src/genfilt.h"
#include <random>
#include <iostream>

TEST_CASE("GPU general filter operations", "[genfilt][gpu]")
{
	srand(time(NULL));

	const std::size_t nKernel = 5;
	const std::size_t range = (nKernel - 1) / 2;
	const std::size_t nx = 500;
	const std::size_t ny = 501;
	const std::size_t nz = 603;

	const std::size_t nxPadded = nx + nKernel - 1;
	const std::size_t nyPadded = ny + nKernel - 1;
	// const std::size_t nzPadded = nz + nKernel - 1;

	vector3<std::size_t> testPos = {40, 41, 45};

	genfilt myFilt;
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

	float* kernelData = new float[myFilt.get_nKernel()];
	for (std::size_t iKernel = 0; iKernel < myFilt.get_nKernel(); iKernel++)
	{
		kernelData[iKernel] = ((float) rand()) / ((float) RAND_MAX);
	}
	myFilt.set_kernel(kernelData);

	// backup an element of the input array
	const float backup_inputData = inputData[testPos.x + nx * (testPos.y + ny * testPos.z)];

	myFilt.conv_gpu();
	INFO("General filtering took " << myFilt.get_tExec() << " ms to execute (GPU)");

	REQUIRE(backup_inputData == inputData[testPos.x + nx * (testPos.y + ny * testPos.z)]);

	float* outputMatrix = myFilt.get_pdataOutput();
	REQUIRE(outputMatrix != nullptr);

	bool allZero = 1;
	
	// check that no value here exceeds the boundaries
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		const float currVal = outputMatrix[iElem];
		REQUIRE(currVal >= 0.0f);

		if (currVal != 0.0f)
			allZero = 0;
	}

	REQUIRE_FALSE(allZero);

	// validate output at a single position
	float testVal = 0;
	std::size_t idxKernel = 0;
	for (std::size_t iz = (testPos.z - range); iz <= (testPos.z + range); iz++) 
	{
		for (std::size_t iy = (testPos.y - range); iy <= (testPos.y + range); iy++) 
		{
			for (std::size_t ix = (testPos.x - range); ix <= (testPos.x + range); ix++) 
			{
				const std::size_t idxVol = ix + nx * (iy + ny * iz); // index of volume
				testVal += (kernelData[idxKernel] * inputData[idxVol]);				
				idxKernel++;
			}
		}
	}
	
	const float valueProc = outputMatrix[testPos.x + nx * (testPos.y + ny * testPos.z)];
	const float errorAmount = fabsf(valueProc - testVal);

	REQUIRE(errorAmount < 1e-5);

	INFO("GPU test passed");

	myFilt.conv();
	INFO("General filtering took " << myFilt.get_tExec() << " ms to execute (CPU)");

	delete[] inputData;
	delete[] kernelData;
}