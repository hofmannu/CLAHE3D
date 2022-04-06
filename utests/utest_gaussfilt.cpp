/*
	File: utest_gaussfilt.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 14.03.2022
*/

#include "../src/gaussfilt.h"

int main()
{

	srand(time(0));
	std::size_t nKernel = 11;
	float sigma = 1.1f;

	gaussfilt myFilt;
	myFilt.set_kernelSize({nKernel, nKernel, nKernel});
	myFilt.set_dataSize({100, 110, 120});
	myFilt.set_sigma(sigma);
	
	float* inputData = new float[myFilt.get_nData()];
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		inputData[iElem] = ((float) rand()) / ((float) RAND_MAX);
	}

	myFilt.set_dataInput(inputData);
	myFilt.run();
	printf("Kernel execution tool %.2f ms\n", myFilt.get_tExec());

	float* outputMatrix = myFilt.get_pdataOutput();

	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		const float currVal = outputMatrix[iElem];
		if (currVal < 0.0)
		{
			printf("in this super simple test case there should be nothing below 0\n");
			throw "InvalidResult";
		}

		if (currVal > 1.00001)
		{
			printf("in this super simple test case there should be nothing above 1.0: %.2f\n",
				currVal);
			throw "InvalidResult";
		}
	}

	// generate an example kernel
	const std::size_t range = (nKernel - 1) / 2;
	float* testKernel = new float [nKernel * nKernel * nKernel];
	for (std::size_t zrel = -range; zrel <= range; zrel++)
	{
		const std::size_t zAbs = range + zrel;
		for (std::size_t yrel = -range; yrel <= range; yrel++)
		{
			const std::size_t yAbs = range + yrel;
			for (std::size_t xrel = -range; xrel <= range; xrel++)
			{
				const std::size_t xAbs = range + xrel;
				const float dr = powf(xrel * xrel + yrel * yrel + zrel * zrel, 0.5f);
				const float gaussval = expf(-1.0f / 2.0f / (dr * dr) / (sigma * sigma))
					/ (sigma * powf(2.0f * M_PI, 0.5f));
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
	float testVal = 0;
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
				testVal = fmaf(testKernel[kernelIdx], inputData[dataIdx], testVal);
			}
		}
	}

	const int idxOutput = testPos.x + 100 * (testPos.y + 110 * testPos.z);
	
	const float errorVal = fabsf(testVal - outputMatrix[idxOutput]);
	if (errorVal >= 1e-6f)
	{
		printf("The test value (%.4f) seems to differ from the class result (%.4f).\n",
			testVal, outputMatrix[idxOutput]);
		throw "InvalidValue";
	}

	delete[] testKernel;
	delete[] inputData;

	return 0;
}