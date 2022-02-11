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

	const float clipLimit = 0.1;
	const uint64_t binSize = 50;

	const uint64_t subVolSize[3] = {30, 30, 30};

	interpGrid histGrid;
	histGrid.setGridSpacing(subVolSize);
	float origin[3];
	for (unsigned char iDim = 0; iDim < 3; iDim++)
		origin[iDim] = 0.5 * (float) subVolSize[iDim];
	
	histGrid.setGridOrigin(origin);
	const uint64_t gridSize[3] = {nZ, nX, nY};
	histGrid.setVolumeSize(gridSize);
	histGrid.calcSubVols();

	histeq histHandler;
	histHandler.setNBins(binSize);
	histHandler.setNoiseLevel(clipLimit);
	histHandler.setVolSize(gridSize);
	histHandler.setSizeSubVols(subVolSize);
	histHandler.setData(inputVol);
	
	printf("[clahe3d] calculating histograms for subvolumes\n");
	histHandler.calculate();

	// loop over full volume
	printf("[clahe3d] running histogram equilization for each voxel\n");

	uint64_t neighbours[6]; // index of next neighbouring elements
	float ratio[3]; // ratios in z x y
	float currValue; // value of position in input volume
	for(uint64_t iY = 0; iY < nY; iY++){
		for (uint64_t iX = 0; iX < nX; iX++){
			for (uint64_t iZ = 0; iZ < nZ; iZ++){
				currValue = inputVol[iZ + nZ * (iX + nX * iY)];
	
				const uint64_t position[3] = {iZ, iX, iY};
				histGrid.getNeighbours(position, neighbours, ratio);
				// printf("Position we have is: %d, %d, %d\n", 
				// 	(int) position[0], (int) position[1], (int) position[2]);
				// printf("Neighbours we have are: %d ... %d, %d ... %d, %d ... %d\n", 
				// 	(int) neighbours[0], (int) neighbours[1], (int) neighbours[2], (int) neighbours[3], (int) neighbours[4], (int) neighbours[5]);
				
				// assign new value based on trilinear interpolation
				inputVol[iZ + nZ * (iX + nX * iY)] =
				// first two opposing z corners
				((histHandler.get_icdf(neighbours[0], neighbours[2], neighbours[4], currValue) * (1 - ratio[0]) + 
				histHandler.get_icdf(neighbours[1], neighbours[2], neighbours[4], currValue) * ratio[0]) 
					* (1 - ratio[1]) +
				// fourth two opposing z corners
				(histHandler.get_icdf(neighbours[0], neighbours[3], neighbours[5], currValue) * (1 - ratio[0]) + 
				histHandler.get_icdf(neighbours[1], neighbours[3], neighbours[5], currValue) * ratio[0])
					* ratio[1]) * (1 - ratio[2]) +
				// second two opposing z corners
				((histHandler.get_icdf(neighbours[0], neighbours[3], neighbours[4], currValue) * (1 - ratio[0]) +
				histHandler.get_icdf(neighbours[1], neighbours[3], neighbours[4], currValue) * ratio[0])
					* (1 - ratio[1]) +
				// third two opposing z corners
				(histHandler.get_icdf(neighbours[0], neighbours[2], neighbours[5], currValue) * (1 - ratio[0]) +
				histHandler.get_icdf(neighbours[1], neighbours[2], neighbours[5], currValue) * ratio[0])
					* ratio[1]) * ratio[2];
			}
		}
	}

	printf("[clahe3d] cleaning up\n");
	delete[] inputVol;
		
	return 0;

}
