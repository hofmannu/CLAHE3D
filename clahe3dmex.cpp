#include <fstream>
#include "interpGrid.h"
#include "histeq.h"
#include <mex.h>

using namespace std;


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	
	float* inputVol = (float *) mxGetData(prhs[0]); // input volume
	uint64_t nZ = mxGetDimensions(prhs[0])[0]; // z dimension
	uint64_t nX = mxGetDimensions(prhs[0])[1]; // x dimension
	uint64_t nY = mxGetDimensions(prhs[0])[2]; // y dimension
	uint64_t * subVolSize = (uint64_t*) mxGetData(prhs[1]); // size of bins [z, x, y]
	float* clipLimit = (float *) mxGetData(prhs[2]);
	uint64_t* binSize = (uint64_t*) mxGetData(prhs[3]);

	printf("[clahe3d] mex code started\n");

	// generate interpolation grid
	printf("[clahe3d] generating interpolation grid\n");
	interpGrid histGrid;
	histGrid.setGridSpacing(subVolSize);
	float origin[3] = {0.5, 0.5, 0.5};
	histGrid.setGridOrigin(origin);
	uint64_t gridSize[3] = {nZ, nX, nY};
	histGrid.setGridSize(gridSize);

	printf("[clahe3d] initializing histogram handler\n");
	histeq histHandler;
	histHandler.setNBins(binSize[0]);
	histHandler.setNoiseLevel(clipLimit[0]);
	histHandler.setVolSize(gridSize);
	histHandler.setSizeSubVols(subVolSize);
	histHandler.setData(inputVol);
	
	printf("[clahe3d] calculating historgrams for subvolumes\n");
	histHandler.calculate();

	// allocate memory for transfer function (each subvol with length of bin)

	// loop over full volume
	printf("[clahe3d] running histogram equilization for each voxel\n");

	uint64_t* position = new uint64_t[3];
	uint64_t* neighbours = new uint64_t [6]; // index of next neighbouring elements
	float* ratio = new float[3]; // ratios in z x y
	float currValue; // value of position in input volume
	uint64_t currNeighbour[3];
	float val[8];
	for(position[2] = 0; position[2] < nY; position[2]++){
		for (position[1] = 0; position[1] < nX; position[1]++){
			for (position[0] = 0; position[0] < nZ; position[0]++){
				currValue = inputVol[position[0] + position[1] * nZ + position[2] * nZ * nX];
				histGrid.getNeighbours(position, neighbours, ratio);
			}
		}
	}

	printf("[clahe3d] cleaning up\n");
	delete[] position;
	delete[] ratio;
	delete[] neighbours;

	return;
}
