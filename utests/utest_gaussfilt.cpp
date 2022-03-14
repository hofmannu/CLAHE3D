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
	int nKernel = 11;
	float sigma = 1.1f;

	gaussfilt myFilt;
	myFilt.set_kernelSize({nKernel, nKernel, nKernel});
	myFilt.set_dataSize({100, 110, 120});
	myFilt.set_sigma(sigma);
	
	float* inputData = new float[myFilt.get_nData()];
	for (int iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		inputData[iElem] = ((float) rand()) / ((float) RAND_MAX);
	}

	myFilt.set_dataInput(inputData);
	myFilt.run();
	printf("Kernel execution tool %.2f ms\n", myFilt.get_tExec());

	float* outputMatrix = myFilt.get_pdataOutput();

	for (int iElem = 0; iElem < myFilt.get_nData(); iElem++)
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
	const int range = (nKernel - 1) / 2;
	float* testKernel = new float [nKernel * nKernel * nKernel];
	for (int zrel = -range; zrel <= range; zrel++)
	{
		const int zAbs = range + zrel;
		for (int yrel = -range; yrel <= range; yrel++)
		{
			const int yAbs = range + yrel;
			for (int xrel = -range; xrel <= range; xrel++)
			{
				const int xAbs = range + xrel;
				const float dr = powf(xrel * xrel + yrel * yrel + zrel * zrel, 0.5f);
				const float gaussval = expf(-1.0f / 2.0f / (dr * dr) / (sigma * sigma))
					/ (sigma * powf(2.0f * M_PI, 0.5f));
				const int dataIdx = xAbs + nKernel * (yAbs + nKernel * zAbs);
				testKernel[dataIdx] = gaussval;
			}
		}
	}

	// normalize kernel to have a total value of 1
	float kernelSum = 0;
	for (int iElem = 0 ; iElem < (nKernel * nKernel * nKernel); iElem++)
		kernelSum += testKernel[iElem];
	
	for (int iElem = 0 ; iElem < (nKernel * nKernel * nKernel); iElem++)
		testKernel[iElem] = testKernel[iElem] / kernelSum;

	// make a small test for an output element
	vector3<int> testPos = {42, 32, 43};
	float testVal = 0;
	for (int zrel = 0; zrel < nKernel; zrel++)
	{
		const int zAbs = testPos.z + zrel - range;
		for (int yrel = 0; yrel < nKernel; yrel++)
		{
			const int yAbs = testPos.y + yrel - range;
			for (int xrel = 0; xrel < nKernel; xrel++)
			{
				const int xAbs = testPos.x + xrel - range;
				const int dataIdx = xAbs + 100 * (yAbs + 110 * zAbs);
				const int kernelIdx = xrel + nKernel * (yrel + nKernel * zrel); 
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