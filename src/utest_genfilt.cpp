/*
	utest_genfilt.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 20.02.2022

	a generic filter class which applies a convolution with a kernel to the dataset
	can be used as a basis for any type of filter
*/

#include "genfilt.h"
#include "vector3.h"

int main()
{
	genfilt myFilt;

	// define the kernel size
	myFilt.set_kernelSize({5, 7, 9});
	vector3<int> range = myFilt.get_range();
	if ((range.x != 2) || (range.y != 3) || (range.z != 4))
	{
		printf("The range calculaiton somehow went wrong\n");
		throw "InvalidResult";
	}

	if (myFilt.get_nKernel() != (5 * 7 * 9))
	{
		printf("The kernel size calculation somehow went wrong\n");
		throw "InvalidResult";
	}

	myFilt.set_dataSize({100, 110, 120});
	vector3<int> paddedSize = myFilt.get_paddedSize();
	if ((paddedSize.x != 104) || (paddedSize.y != 116) || (paddedSize.z != 128))
	{
		printf("Padded size calculation went wrong\n");
		throw "InvalidResult";
	}

	// generate an input array of 1s
	float* inputData = new float[myFilt.get_nData()];
	for (int iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		inputData[iElem] = 1;
	}

	// generate a kernel array of 1s
	float* kernel = new float[myFilt.get_nKernel()];
	for (int iElem = 0; iElem < myFilt.get_nKernel(); iElem++)
	{
		kernel[iElem] = 1;
	}	

	myFilt.set_dataInput(inputData);
	myFilt.set_kernel(kernel);

	myFilt.conv();

	// in this case all values should be between 0 and nKernel
	float* outputMatrix = myFilt.get_pdataOutput();
	for (int iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		const float currVal = outputMatrix[iElem];
		if (currVal < 0.0)
		{
			printf("in this super simple test case there should be nothing below 0\n");
			throw "InvalidResult";
		}

		if (currVal > myFilt.get_nKernel())
		{
			printf("in this super simple test case there should be nothing above nKernel\n");
			throw "InvalidResult";
		}
	}


	delete[] inputData;
	return 0;
}