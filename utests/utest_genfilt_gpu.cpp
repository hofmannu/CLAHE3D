/*
	File: utest_genfilt_gpu.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 01.04.2022

	Description: runs the general filtering algorithm on the GPU and checks for correctness
*/


#include "../src/genfilt.h"
#include <random>
#include <iostream>

int main()
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

	float* kernelData = new float[myFilt.get_nKernel()];
	for (std::size_t iKernel = 0; iKernel < myFilt.get_nKernel(); iKernel++)
	{
		kernelData[iKernel] = ((float) rand()) / ((float) RAND_MAX);
	}
	myFilt.set_kernel(kernelData);

	// backup an element of the input array
	const float backup_inputData = inputData[testPos.x + nx * (testPos.y + ny * testPos.z)];

	myFilt.conv_gpu();
	printf("General filtering took %.2f ms to execute (GPU)\n", myFilt.get_tExec());

	if (backup_inputData != inputData[testPos.x + nx * (testPos.y + ny * testPos.z)])
	{
		printf("gen filtering somehow altered the input data, should not happen!\n");
		throw "InvalidValue";
	}

	float* outputMatrix = myFilt.get_pdataOutput();
	if (outputMatrix == nullptr)
	{
		printf("Filtering the dataset resulted in a null pointer matrix\n");
		throw "InvalidValue";
	}

	bool allZero = 1;
	
	// check that no value here exceeds the boundaries
	for (std::size_t iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		const float currVal = outputMatrix[iElem];
		if (currVal < 0.0f)
		{
			printf("in this super simple test case there should be nothing below 0\n");
			throw "InvalidResult";
		}


		if (currVal != 0.0f)
			allZero = 0;
	}

	if (allZero)
	{
		printf("Looks like the calculation returend an only zeros array\n");
		throw "InvalidValue";
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

	if (errorAmount > 1e-5)
	{
		printf("Comparison results differ: CPU = %.4f, GPU = %.4f\n", testVal, valueProc);
		throw "InvalidValue";
	}

	printf("Everything passed just fine\n");

	myFilt.conv();
	printf("General filtering took %.2f ms to execute (CPU)\n", myFilt.get_tExec());


	delete[] inputData;
	delete[] kernelData;




	return 0;
}