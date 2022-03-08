/*
	unit test used to check if our cdf is producing the same result on cpu and gpu
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include "../src/histeq.h"
#include <iostream>
#include <cstdint>
#include <fstream>
#include <chrono>
#include "../src/vector3.h"

using namespace std;

int main(){

	srand(1);
	// define grid dimensions for testing
	// initialize some parameters
	const float clipLimit = 0.1;
	const int binSize = 10;
	const vector3<int> subVolSpacing = {20, 20, 20};
	const vector3<int> volSize = {600, 500, 400};
	const vector3<int> subVolSize(11, 11, 11);

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[volSize[0] * volSize[1] * volSize[2]];
	for(int iIdx = 0; iIdx < (volSize[0] * volSize[1] * volSize[2]); iIdx ++)
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);
		// this should generate a random number between 0 and 1


	const int iBin = rand() % binSize;

	histeq histHandler;
	histHandler.set_nBins(binSize);
	histHandler.set_noiseLevel(clipLimit);
	histHandler.set_volSize(volSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);

	// histogram calculation on GPU
	histHandler.calculate_cdf_gpu();
	
	// backup the version of the CDF calculated with
	float* cdf_bk = new float[histHandler.get_ncdf()];
	for (int iElem = 0; iElem < histHandler.get_ncdf(); iElem++)
	{
		cdf_bk[iElem] = histHandler.get_cdf(iElem);
	}


	// histogram calculation of CPU
	histHandler.calculate_cdf();
	bool isSame = 1;
	int countNotSame = 0;
	for (int iElem = 0; iElem < histHandler.get_ncdf(); iElem++)
	{
		const float deltaVal = abs(cdf_bk[iElem] - histHandler.get_cdf(iElem));
		if (cdf_bk[iElem] != histHandler.get_cdf(iElem))
		{
			isSame = 0;
			countNotSame++;
		}
	}

	printf("Displaying example CDF function:\n");
	for (int iBin = 0; iBin < binSize; iBin++)
	{
		printf("iBin: %d, GPU: %.3f, CPU: %.3f\n", 
			(int) iBin, cdf_bk[iBin + 20], histHandler.get_cdf(iBin + 20));
	}

	// compare if results are the same
	if (!isSame)
	{
		const float percOff = ((float) countNotSame / ((float) histHandler.get_ncdf())) * 100.0;
		printf("CPU and GPU results differ for %.1f percent of the elements\n", percOff);
		throw "InvalidResult";
	}
	else
	{
		printf("GPU and CPU deliver the same result for CDF!\n");
	}
	
	delete[] inputVol;
	delete[] cdf_bk;
		
	return 0;

}
