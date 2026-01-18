
/* 
	a simple test validating the performance of the meanfilter
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 10.03.2022
*/

#include <catch2/catch.hpp>
#include "../src/meanfilt.h"

TEST_CASE("mean filter operations", "[meanfilt]")
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
		REQUIRE(currVal >= 0.0);
		REQUIRE(currVal <= 1.00001);
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

	REQUIRE(errorVal < 1e-6);
}