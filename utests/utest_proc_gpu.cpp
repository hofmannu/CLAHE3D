/*
	Simple runtime test to check if any errors occur during execution (no results checked)
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

int main()
{

	// define grid dimensions for testing
	const vector3<std::size_t> volSize(600, 500, 400);
	const float clipLimit = 0.1;
	const int binSize = 20;
	const vector3<std::size_t> subVolSize(31, 31, 31);
	const vector3<std::size_t> subVolSpacing(20, 20, 20);

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[volSize.elementMult()];
	for(int iIdx = 0; iIdx < (volSize.elementMult()); iIdx ++)
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);
		// this should generate a random number between 0 and 1

	// initialize some parameters

	histeq histHandler;
	histHandler.set_nBins(binSize);
	histHandler.set_noiseLevel(clipLimit);
	histHandler.set_volSize(volSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	
	// histogram calculation on GPU
	histHandler.calculate_cdf_gpu();
	histHandler.equalize_gpu();

	delete[] inputVol;
		
	return 0;

}