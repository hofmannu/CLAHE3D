// File: test.cpp
// Function independent from matlab libraries used to debug
// since MATLAB debugging is a pain inn the ass
//
// Author: Urs Hofmann
// Mail: hofmannu@biomed.ee.ethz.ch
// Date: 23.11.2019
// Version: 1.0

#include "histeq.h"
#include "interpGrid.h"
#include <iostream>
#include <cstdint>
#include <fstream>

using namespace std;

int main(){

	printf("Testing CLAHE3D code functionality\n");

	// define grid dimensions for testing
	const uint64_t nZ = 600;
 	const uint64_t nX = 500;
	const uint64_t nY = 400;

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[nX * nY * nZ];
	for(uint64_t iIdx = 0; iIdx < (nX * nY * nZ); iIdx ++)
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);
		// this should generate a random number between 0 and 1

	// initialize some parameters
	const float clipLimit = 0.1;
	const uint64_t binSize = 250;
	const uint64_t subVolSize[3] = {31, 31, 31};
	const uint64_t subVolSpacing[3] = {20, 20, 20};
	const uint64_t gridSize[3] = {nZ, nX, nY};

	histeq histHandler;
	histHandler.setNBins(binSize);
	histHandler.setNoiseLevel(clipLimit);
	histHandler.setVolSize(gridSize);
	histHandler.setSizeSubVols(subVolSize);
	histHandler.setSpacingSubVols(subVolSpacing);
	histHandler.setData(inputVol);
	
	printf("[clahe3d] calculating histograms for subvolumes\n");
	histHandler.calculate();

	// loop over full volume
	printf("[clahe3d] running histogram equilization for each voxel\n");
	histHandler.equalize();
	
	delete[] inputVol;
		
	return 0;

}
