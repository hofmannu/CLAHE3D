// File: test.cpp
// Function independent from matlab libraries used to debug
// since MATLAB debugging is a pain inn the ass
//
// Author: Urs Hofmann
// Mail: hofmannu@biomed.ee.ethz.ch
// Date: 23.11.2019
// Version: 1.0

#include "histeq.h"
#include <iostream>
#include <cstdint>
#include <fstream>
#include <chrono>
#include "vector3.h"

using namespace std;

int main()
{

	// parameters
	const vector3<int> volSize(600, 500, 300);
	const float clipLimit = 0.1;
	const int binSize = 250;
	const vector3<int> subVolSize(31, 31, 31);
	const vector3<int> subVolSpacing(20, 20, 20);

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[volSize.x * volSize.y * volSize.z];
	for(int iIdx = 0; iIdx < (volSize.x * volSize.y * volSize.z); iIdx ++)
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);
		// this should generate a random number between 0 and 1

	histeq histHandler;
	histHandler.set_nBins(binSize);
	histHandler.set_noiseLevel(clipLimit);
	histHandler.set_volSize(volSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	
	// histogram calculation on GPU
	histHandler.calculate_cdf();
	// histHandler.calculate_cdf_gpu();
	
	histHandler.equalize();
	// histHandler.equalize_gpu();
	delete[] inputVol;
		
	return 0;

}
