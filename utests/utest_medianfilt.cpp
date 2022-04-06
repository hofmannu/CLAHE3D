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
	srand(time(NULL));

	const std::size_t nKernel = 5;
	const std::size_t range = (nKernel - 1) / 2;
	const std::size_t nx = 100;
	const std::size_t ny = 110;
	const std::size_t nz = 120;

	medianfilt myFilt;
	
	// set the size of the kernel and check if everything went smooth
	myFilt.set_kernelSize({nKernel, nKernel, nKernel});
	if (myFilt.get_nKernel() != (nKernel * nKernel * nKernel))
	{
		printf("Something went wrong while setting the kernel size\n");
		throw "InvalidValue";
	}

	// set the size of the dataset and check if it was successful
	myFilt.set_dataSize({nx, ny, nz});
	if (myFilt.get_nData() != (nx * ny * nz))
	{
		printf("Something went wrong while defining the data size\n");
		throw "InvalidValue";
	}

	// generate a volume with random values
	float* inputData = new float[myFilt.get_nData()];
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		inputData[iElem] = ((float) rand()) / ((float) RAND_MAX);
	}
	myFilt.set_dataInput(inputData);

	const float backupVal = inputData[101];

	myFilt.run();
	printf("Median filtering took %.2f sec to execute\n", myFilt.get_tExec() * 1e-3f);

	// check if input matrix remained the same
	if (backupVal != inputData[101])
	{
		printf("Median filtering modified our input matrix, this should not happen.\n");
		throw "InvalidValue";
	}

	float* outputMatrix = myFilt.get_pdataOutput();
	
	// check that no value here exceeds the boundaries
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		const float currVal = outputMatrix[iElem];
		if (currVal < 0.0f)
		{
			printf("in this super simple test case there should be nothing below 0\n");
			throw "InvalidResult";
		}

		if (currVal > 1.0f)
		{
			printf("in this super simple test case there should be nothing above 1.0: %.2f\n",
				currVal);
			throw "InvalidResult";
		}
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

	if (valueProc != testVal)
	{
		printf("Comparison results differ: %.4f, %.4f\n", testVal, valueProc);
		throw "InvalidValue";
	}

	delete[] inputData;

	return 0;
}