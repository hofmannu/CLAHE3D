
/* 
	a simple test validating the performance of the meanfilter
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 10.03.2022
*/

#include "../src/meanfilt.h"

int main()
{
	const int nKernel = 11;
	const int range = (nKernel - 1) / 2;

	meanfilt myFilt;
	myFilt.set_kernelSize({nKernel, nKernel, nKernel});
	myFilt.set_dataSize({100, 110, 120});
	
	float* inputData = new float[myFilt.get_nData()];
	for (int iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		inputData[iElem] = ((float) rand()) / ((float) RAND_MAX);
	}
	myFilt.set_dataInput(inputData);
	myFilt.run();


	float* outputMatrix = myFilt.get_pdataOutput();
	
	// check that no value here exceeds the boundaries
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

	// validate mean at a single position
	float testMean = 0;
	vector3<int> testPos = {40, 40, 40};
	for (int iz = testPos.z - range; iz <= testPos.z + range; iz++) 
	{
		for (int iy = testPos.y - range; iy <= testPos.y + range; iy++) 
		{
			for (int ix = testPos.x - range; ix <= testPos.x + range; ix++) 
			{
				int idx = ix + 100 * (iy + 110 * iz);
				testMean += inputData[idx];
			}
		}
	}
	testMean /= ((float) nKernel * (float) nKernel * (float) nKernel);

	const float valueProc = outputMatrix[testPos.x + 100 * (testPos.y + 110 * testPos.z)];
	const float errorVal = fabsf(testMean - valueProc);

	if (errorVal > 1e-6)
	{
		printf("Comparison results differ: %.4f, %.4f\n", testMean, valueProc);
		throw "InvalidValue";
	}
	
	return 0;
}