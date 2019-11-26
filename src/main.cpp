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

	printf("Testing CLAHE3D code fuctioality\n");

	// define grid dimensions for testing
	uint64_t nZ = 600;
 	uint64_t nX = 500;
	uint64_t nY = 400;

	// generate input volume matrix and assign random values to it
	float* inputVol = new float[nX * nY * nZ];
	for(uint64_t iIdx = 0; iIdx < (nX * nY * nZ); iIdx ++)
		inputVol[iIdx] = ((float) (iIdx % 200)) * 0.1;

	float clipLimit = 0.5;
	uint64_t binSize = 50;

	uint64_t* subVolSize = new uint64_t[3];
	subVolSize[0] = 25;
	subVolSize[1] = 25;
	subVolSize[2] = 25;

	interpGrid histGrid;
	histGrid.setGridSpacing(subVolSize);
	float origin[3];
	for (unsigned char iDim = 0; iDim < 3; iDim++)
		origin[iDim] = 0.5 * (float) subVolSize[iDim];
	
	histGrid.setGridOrigin(origin);
	uint64_t gridSize[3] = {nZ, nX, nY};
	histGrid.setGridSize(gridSize);

	histeq histHandler;
	histHandler.setNBins(binSize);
	histHandler.setNoiseLevel(clipLimit);
	histHandler.setVolSize(gridSize);
	histHandler.setSizeSubVols(subVolSize);
	histHandler.setData(inputVol);
	
	printf("[clahe3d] calculating historgrams for subvolumes\n");
	histHandler.calculate();

	// loop over full volume
	printf("[clahe3d] running histogram equilization for each voxel\n");

	uint64_t* position = new uint64_t[3];
	uint64_t* neighbours = new uint64_t[6]; // index of next neighbouring elements
	float* ratio = new float[3]; // ratios in z x y
	float currValue; // value of position in input volume
	for(position[2] = 0; position[2] < nY; position[2]++){
		for (position[1] = 0; position[1] < nX; position[1]++){
			for (position[0] = 0; position[0] < nZ; position[0]++){
				currValue = inputVol[position[0] + position[1] * nZ + position[2] * nZ * nX];
				histGrid.getNeighbours(position, neighbours, ratio);
				
				printf("%f ", inputVol[position[0] + position[1] * nZ + position[2] * nZ * nX]);
				for (unsigned char iPos = 0; iPos < 6; iPos++)
					printf("%ld, ", neighbours[iPos]);
				
						
				inputVol[position[0] + position[1] * nZ + position[2] * nZ * nX] =
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
				
				printf("%f\n", inputVol[position[0] + position[1] * nZ + position[2] * nZ * nX]);
			}
		}
	}

	printf("[clahe3d] cleaning up\n");
	delete[] position;
	delete[] ratio;
	delete[] neighbours;
	delete[] inputVol;
	delete[] subVolSize;
		
	return 0;
	
	
	return 0;
}
