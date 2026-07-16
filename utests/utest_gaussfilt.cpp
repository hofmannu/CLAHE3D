/*
	File: utest_gaussfilt.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 14.03.2022
*/

#include <catch2/catch_test_macros.hpp>

#include "../src/gaussfilt.h"

TEST_CASE("gaussfilt stays in bounds and matches a hand-computed convolution", "[gaussfilt]")
{
	srand(0);
	constexpr std::size_t nKernel = 11;
	constexpr float sigma = 1.1f;

	gaussfilt myFilt;
	myFilt.set_kernelSize({nKernel, nKernel, nKernel});
	myFilt.set_dataSize({100, 110, 120});
	myFilt.set_sigma(sigma);

	std::vector<float> inputData(myFilt.get_nData());
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		inputData[iElem] = ((float) rand()) / ((float) RAND_MAX);
	}

	myFilt.set_dataInput(inputData.data());
	myFilt.run();

	float* outputMatrix = myFilt.get_pdataOutput();

	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		const float currVal = outputMatrix[iElem];
		REQUIRE(currVal >= 0.0f);
		REQUIRE(currVal <= 1.00001f);
	}

	// generate an example kernel, mirroring gaussfilt::run() exactly (note the
	// relative offsets must be signed here: the class centres the kernel with
	// dz = iz - range, so a std::size_t loop would underflow and never run)
	const std::size_t range = (nKernel - 1) / 2;
	std::vector<float> testKernel(nKernel * nKernel * nKernel);
	for (std::size_t zAbs = 0; zAbs < nKernel; zAbs++)
	{
		const float dz = (float) zAbs - (float) range;
		for (std::size_t yAbs = 0; yAbs < nKernel; yAbs++)
		{
			const float dy = (float) yAbs - (float) range;
			for (std::size_t xAbs = 0; xAbs < nKernel; xAbs++)
			{
				const float dx = (float) xAbs - (float) range;
				const float dr = powf(dx * dx + dy * dy + dz * dz, 0.5f);
				const float gaussval = expf(-(dr * dr) / (2.0f * sigma * sigma));
				const std::size_t dataIdx = xAbs + nKernel * (yAbs + nKernel * zAbs);
				testKernel[dataIdx] = gaussval;
			}
		}
	}

	// normalize kernel to have a total value of 1
	float kernelSum = 0;
	for (std::size_t iElem = 0 ; iElem < (nKernel * nKernel * nKernel); iElem++)
		kernelSum += testKernel[iElem];

	for (std::size_t iElem = 0 ; iElem < (nKernel * nKernel * nKernel); iElem++)
		testKernel[iElem] = testKernel[iElem] / kernelSum;

	// make a small test for an output element
	vector3<std::size_t> testPos = {42, 32, 43};
	double testVal = 0.0;
	for (std::size_t zrel = 0; zrel < nKernel; zrel++)
	{
		const std::size_t zAbs = testPos.z + zrel - range;
		for (std::size_t yrel = 0; yrel < nKernel; yrel++)
		{
			const std::size_t yAbs = testPos.y + yrel - range;
			for (std::size_t xrel = 0; xrel < nKernel; xrel++)
			{
				const std::size_t xAbs = testPos.x + xrel - range;
				const std::size_t dataIdx = xAbs + 100 * (yAbs + 110 * zAbs);
				const std::size_t kernelIdx = xrel + nKernel * (yrel + nKernel * zrel);
				testVal = fma(testKernel[kernelIdx], inputData[dataIdx], testVal);
			}
		}
	}

	const int idxOutput = testPos.x + 100 * (testPos.y + 110 * testPos.z);

	const float relativeError = fabs(testVal - outputMatrix[idxOutput]) / fabs(testVal);
	INFO("reference value = " << testVal << ", filter result = " << outputMatrix[idxOutput]);
	REQUIRE(relativeError < 1e-4f);
}

TEST_CASE("gaussfilt impulse response peaks at the centre", "[gaussfilt]")
{
	// regression: the old kernel formula exp(-1/(2 r^2 sigma^2)) evaluated to 0 at the
	// centre tap (r == 0 -> exp(-inf)), so an impulse smoothed to 0 exactly where it
	// should peak. A real Gaussian has its maximum weight at the centre.
	constexpr std::size_t nKernel = 9;
	const vector3<std::size_t> dataSize = {21, 21, 21};

	gaussfilt myFilt;
	myFilt.set_kernelSize({nKernel, nKernel, nKernel});
	myFilt.set_dataSize(dataSize);
	myFilt.set_sigma(1.5f);

	std::vector<float> inputData(myFilt.get_nData(), 0.0f);
	const vector3<std::size_t> centre = {10, 10, 10};
	const std::size_t idxCentre = centre.x + dataSize.x * (centre.y + dataSize.y * centre.z);
	inputData[idxCentre] = 1.0f; // single bright voxel

	myFilt.set_dataInput(inputData.data());
	myFilt.run();

	float* out = myFilt.get_pdataOutput();

	// the smoothed impulse must be positive at the centre and be the global maximum
	REQUIRE(out[idxCentre] > 0.0f);
	for (std::size_t i = 0; i < myFilt.get_nData(); i++)
		REQUIRE(out[i] <= out[idxCentre]);
}
