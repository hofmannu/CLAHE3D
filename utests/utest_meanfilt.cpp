#include "../src/meanfilt.h"

int main()
{
	meanfilt myFilt;
	myFilt.set_kernelSize({11, 11, 11});
	myFilt.set_dataSize({100, 110, 120});
	
	float* inputData = new float[myFilt.get_nData()];
	for (int iElem = 0; iElem < myFilt.get_nData(); iElem++)
	{
		inputData[iElem] = 1;
	}
	myFilt.set_dataInput(inputData);
	myFilt.run();

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

	return 0;
}