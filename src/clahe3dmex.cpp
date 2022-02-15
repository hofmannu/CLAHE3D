/*
	File: clahe3dmex.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 11.02.2022
*/

#include <fstream>
#include "histeq.h"
#include <mex.h>

#define USE_CUDA 0

using namespace std;

// clahe3dmex(interpVol, subVolSize, spacingSubVols, clipLimit, binSize);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	
	float* inputVol = (float *) mxGetData(prhs[0]); // input volume
	const int64_t nZ = mxGetDimensions(prhs[0])[0]; // z dimension
	const int64_t nX = mxGetDimensions(prhs[0])[1]; // x dimension
	const int64_t nY = mxGetDimensions(prhs[0])[2]; // y dimension
	int64_t * subVolSize = (int64_t*) mxGetData(prhs[1]); // size of bins [z, x, y]
	int64_t * subVolSpacing = (int64_t*) mxGetData(prhs[2]); // size of bins [z, x, y]
	float* clipLimit = (float *) mxGetData(prhs[3]);
	int64_t* binSize = (int64_t*) mxGetData(prhs[4]);
	int64_t volumeSize[3] = {nZ, nX, nY};

	printf("[clahe3d] initializing histogram handler\n");
	histeq histHandler;
	histHandler.set_nBins(binSize[0]);
	histHandler.set_noiseLevel(clipLimit[0]);
	histHandler.set_volSize(volumeSize);
	histHandler.set_sizeSubVols(subVolSize);
	histHandler.set_spacingSubVols(subVolSpacing);
	histHandler.set_data(inputVol);
	histHandler.set_overwrite(1);

	printf("[clahe3d] calculating historgrams for subvolumes\n");
	histHandler.calculate_cdf();

	// loop over full volume
	printf("[clahe3d] running histogram equilization for each voxel\n");
	histHandler.equalize();

	return;
}
