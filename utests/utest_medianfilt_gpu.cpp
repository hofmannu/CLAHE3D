/* 
	File: utest_medianfilt_gpu.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 31.03.2022

	Description: Runs a median filter over volume on the GPU

*/

#include <iostream>
#include "../src/medianfilt.h"
#include <random>

int main()
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
	if (myFilt.get_nKernel() != (5 * 5 * 5))
	{
		printf("Something went wrong while defining the kernel size\n");
		throw "InvalidValue";
	}

	myFilt.set_dataSize({nx, ny, nz});
	if (myFilt.get_nData() != (nx * ny * nz))
	{
		printf("Something went wrong while setting the dataset size\n");
		throw "InvalidValue";
	}

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
	printf("Median filtering took %.2f ms to execute (GPU)\n", myFilt.get_tExec());

	if (backup_inputData != inputData[testPos.x + nx * (testPos.y + ny * testPos.z)])
	{
		printf("Median filtering somehow altered the input data, should not happen!\n");
		throw "InvalidValue";
	}

	float* outputMatrix = myFilt.get_pdataOutput();
	if (outputMatrix == nullptr)
	{
		printf("Filtering the dataset resulted in a null pointer matrix\n");
		throw "InvalidValue";
	}
	
	// check that no value here exceeds the boundaries
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

	// validate output from padded array
	// float * paddedData = myFilt.get_pdataPadded();
	// for (std::size_t iz = 0; iz < nKernel; iz++) 
	// {
	// 	const std::size_t zAbs = iz + testPos.z;
	// 	for (std::size_t iy = 0; iy < nKernel; iy++) 
	// 	{
	// 		const std::size_t yAbs = iy + testPos.y;
	// 		for (std::size_t ix = 0; ix < nKernel; ix++) 
	// 		{
	// 			const std::size_t xAbs = ix + testPos.x;
	// 			const std::size_t idxVol = xAbs + nxPadded * (yAbs + nyPadded * zAbs); // index of volume
	// 			printf("(%lu, %lu, %lu): %.2f, ", xAbs - range, yAbs - range, zAbs - range, paddedData[idxVol]);
	// 		}
	// 		printf("\n");

	// 	}
	// }
	// printf("\n");

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

	if (valueProc != testVal)
	{
		printf("Comparison results differ: CPU = %.4f, GPU = %.4f\n", testVal, valueProc);
		throw "InvalidValue";
	}

	printf("Everything passed just fine\n");

	myFilt.run();
	printf("Median filtering took %.2f ms to execute (CPU)\n", myFilt.get_tExec());


	delete[] inputData;


	return 0;
}