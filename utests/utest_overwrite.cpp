/*
	overwrite test: checks if the overwrite function is working properly
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include "../src/histeq.h"
#include <iostream>
#include <cstdint>
#include <fstream>
#include <chrono>

using namespace std;

int main(){

	const vector3<int> volSize(600, 500, 300);
	const float clipLimit = 0.1;
	const int binSize = 250;
	const vector3<int> subVolSize(31, 31, 31);
	const vector3<int> subVolSpacing(20, 20, 20);
	

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[volSize.x * volSize.y * volSize.z];
	float* inputVolBk = new float[volSize.x * volSize.y * volSize.z];
	for(int iIdx = 0; iIdx < (volSize.x * volSize.y * volSize.z); iIdx ++)
	{
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);
		inputVolBk[iIdx] = inputVol[iIdx];
	}
		// this should generate a random number between 0 and 1

	histeq histHandler;
	histHandler.set_nBins(binSize);
	histHandler.set_noiseLevel(clipLimit);
	histHandler.set_volSize(volSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	histHandler.set_overwrite(0);
	
	// histogram calculation on GPU
	histHandler.calculate_cdf();
	histHandler.equalize();

	// check if input volume remained the same
	for (int iElem = 0; iElem < (volSize.x * volSize.y * volSize.z); iElem++)
	{
		if (inputVol[iElem] != inputVolBk[iElem])
		{
			printf("The input volume changed! Not acceptable.\n");
			throw "InvalidBehaviour";
		}
	}

	const float testVal1 = histHandler.get_cdf(120, 15, 2, 5);

	histHandler.equalize();

	const float testVal2 = histHandler.get_cdf(120, 15, 2, 5);

	if (testVal1 != testVal2)
	{
		printf("Test values are not identical!\n");
		throw "InvalidResult";
	}
	else
	{
		printf("Overwrite flag seems to work as expected!\n");
	}

	delete[] inputVol;
	delete[] inputVolBk;
		
	return 0;

}
