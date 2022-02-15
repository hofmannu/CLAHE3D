/* 
	test how our class handles a binarization task
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include "histeq.h"
#include <iostream>
#include <cstdint>
#include <fstream>
#include <chrono>
#include "vector3.h"

using namespace std;

int main()
{

	// define grid dimensions for testing
	const vector3<int64_t> volSize(600, 500, 400);
	const float clipLimit = 0.1;
	const int64_t binSize = 250;
	const vector3<int64_t> subVolSize(31, 31, 31);
	const vector3<int64_t> subVolSpacing(20, 20, 20);

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[volSize.elementMult()];
	for(int64_t iIdx = 0; iIdx < volSize.elementMult(); iIdx ++)
	{
		inputVol[iIdx] = ((float) (iIdx % 2)) * 99.0 + 1.0;
	}
		
	// now our whole matrix is either 100 or 1

	// initialize some parameters

	histeq histHandler;
	histHandler.set_nBins(binSize);
	histHandler.set_noiseLevel(clipLimit);
	histHandler.set_volSize(volSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	histHandler.set_overwrite(0);
	
	histHandler.calculate_cdf();

	// check the min value in a single example bin
	if (!(histHandler.get_minValBin(0, 0, 0) == 1.0))
	{
		printf("Minimum in each and every bin should be 1.0\n");
		throw "InvalidValue";
	}

	// check maximum value in an example bin
	if (!(histHandler.get_maxValBin(0, 0, 0) == 100.0))
	{
		printf("Minimum in each and every bin should be 100.0\n");
		throw "InvalidValue";
	}

	// check if CDF is valid
	// all bins until last one should have value 0.5 in CDF
	for (int64_t iBin = 0; iBin < binSize; iBin++)
	{
		if (iBin < (binSize - 1))
		{
			if (histHandler.get_cdf(iBin, 0, 0, 0) != 0.0)
			{
				throw "InvalidValue";
			}
		}
		else
		{
			if (histHandler.get_cdf(iBin, 0, 0, 0) != 1.0)
			{
				throw "InvalidValue";
			}
		}
	}

	// check example ICDFs
	// value 1.0 should return bin 0
	// value 

	histHandler.equalize();

	// the output should now be all either 1s or 0s with an even distribution
	float* outputVolCpu = histHandler.get_ptrOutput();
	int64_t counterZero = 0;
	int64_t counterOne = 0;
	for (int64_t iElem = 0; iElem < (volSize.elementMult()); iElem++)
	{
		if (outputVolCpu[iElem] == 0.0)
		{
			counterZero ++;
		}
		else if (outputVolCpu[iElem] == 1.0)
		{
			counterOne ++;
		}
		else
		{
			printf("All elements 0 or 1, most recent: input = %.1f, output = %.1f\n",
				inputVol[iElem], outputVolCpu[iElem]);
			throw "InvalidValue";
		}
	}

	if (counterZero != counterOne)
	{
		printf("Does not look like an even distibution to me.");
		throw "InvalidValue";
	}
	else
	{
		printf("All seems to work, nZeros: %d, nOnes: %d\n", (int)counterZero, (int)counterOne);
	}

	delete[] inputVol;
		
	return 0;

}