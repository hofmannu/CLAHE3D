/*
	a simple test validating the performance of our median filter
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 10.03.2022
*/

#include "../src/medianfilt.h"
#include <vector>
#include <algorithm>

int main()
{
	const int nKernel = 5;
	const int range = (nKernel - 1) / 2;

	medianfilt myFilt;


	myFilt.set_kernelSize({nKernel, nKernel, nKernel});
	myFilt.set_dataSize({100, 110, 120});

	// generate a volume with random values
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

	// validate median at a single position
	std::vector<float> tempArray;
	
	vector3<int> testPos = {40, 40, 40};
	for (int iz = testPos.z - range; iz <= testPos.z + range; iz++) 
	{
		for (int iy = testPos.y - range; iy <= testPos.y + range; iy++) 
		{
			for (int ix = testPos.x - range; ix <= testPos.x + range; ix++) 
			{
				const int idxVol = ix + 100 * (iy + 110 * iz); // index of volume
				tempArray.push_back(inputData[idxVol]);
			}
		}
	}
	sort(tempArray.begin(), tempArray.end());
	const int medianIdx = (nKernel * nKernel * nKernel - 1) / 2;

	const float valueProc = outputMatrix[testPos.x + 100 * (testPos.y + 110 * testPos.z)];
	const float testVal = tempArray[medianIdx];

	if (valueProc != testVal)
	{
		printf("Comparison results differ: %.4f, %.4f\n", testVal, valueProc);
		throw "InvalidValue";
	}

	delete[] inputData;

	return 0;
}