/*
	File: utest_medianfilt_gpu.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 31.03.2022

	Description: Runs a median filter over volume on the GPU
*/

#include <catch2/catch_test_macros.hpp>

#include "../src/medianfilt.h"
#include <random>
#include <algorithm>

TEST_CASE("medianfilt GPU preserves input, stays in bounds and matches a hand-computed median", "[medianfilt][gpu]")
{
	srand(time(NULL));

	const std::size_t nKernel = 5;
	const std::size_t range = (nKernel - 1) / 2;

	const std::size_t nx = 500;
	const std::size_t ny = 510;
	const std::size_t nz = 520;

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

	// the filter must not touch its input
	REQUIRE(backup_inputData == inputData[testPos.x + nx * (testPos.y + ny * testPos.z)]);

	float* outputMatrix = myFilt.get_pdataOutput();
	REQUIRE(outputMatrix != nullptr);

	// check that no value here exceeds the boundaries
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		const float currVal = outputMatrix[iElem];
		REQUIRE(currVal >= 0.0f);
		REQUIRE(currVal <= 1.00001f);
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

	INFO("reference median (CPU) = " << testVal << ", GPU result = " << valueProc);
	REQUIRE(valueProc == testVal);

	// the CPU path should still run afterwards
	REQUIRE_NOTHROW(myFilt.run());

	delete[] inputData;
}

TEST_CASE("medianfilt GPU handles duplicate values (constant volume)", "[medianfilt][gpu]")
{
	// regression guard: kth_smallest carried an "all elements distinct" assumption and
	// an INT_MAX fallback. Median-filter input routinely has duplicates (a constant
	// volume is all duplicates); the interior median must equal the constant, never a
	// ~2.1e9 sentinel.
	const std::size_t nKernel = 3;
	const std::size_t nx = 8, ny = 8, nz = 8;
	const float constVal = 0.7f;

	medianfilt myFilt;
	myFilt.set_kernelSize({nKernel, nKernel, nKernel});
	myFilt.set_dataSize({nx, ny, nz});

	std::vector<float> inputData(myFilt.get_nData(), constVal);
	myFilt.set_dataInput(inputData.data());

	myFilt.run_gpu();
	float* out = myFilt.get_pdataOutput();

	// an interior voxel has an all-constant neighbourhood -> median == constVal
	const std::size_t idxInterior = 4 + nx * (4 + ny * 4);
	REQUIRE(out[idxInterior] == constVal);
}
