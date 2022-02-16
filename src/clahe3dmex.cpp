/*
	File: clahe3dmex.cpp
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 11.02.2022
*/

#include <fstream>
#include "histeq.h"
#include <mex.h>
#include "vector3.h"

using namespace std;

// clahe3dmex(interpVol, subVolSize, spacingSubVols, clipLimit, binSize);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	
	float* inputVol = (float *) mxGetData(prhs[0]); // input volume
	const int nX = mxGetDimensions(prhs[0])[0]; // z dimension
	const int nY = mxGetDimensions(prhs[0])[1]; // x dimension
	const int nZ = mxGetDimensions(prhs[0])[2]; // y dimension
	int * subVolSize = (int*) mxGetData(prhs[1]); // size of bins [z, x, y]
	int * subVolSpacing = (int*) mxGetData(prhs[2]); // size of bins [z, x, y]
	float* clipLimit = (float *) mxGetData(prhs[3]);
	int* binSize = (int*) mxGetData(prhs[4]);

	vector3<int> volumeSize = {nZ, nX, nY};
	vector3<int> subVolSizeVec = {subVolSize[0], subVolSize[1], subVolSize[2]};
	vector3<int> subVolSpacingVec = {subVolSpacing[0], subVolSpacing[1], subVolSpacing[2]};

	printf("[clahe3d] initializing histogram handler\n");
	histeq histHandler;
	histHandler.set_nBins(binSize[0]);
	histHandler.set_noiseLevel(clipLimit[0]);
	histHandler.set_volSize(volumeSize);
	histHandler.set_sizeSubVols(subVolSizeVec);
	histHandler.set_spacingSubVols(subVolSpacingVec);
	histHandler.set_data(inputVol);
	histHandler.set_overwrite(1);

	printf("[clahe3d] calculating historgrams for subvolumes\n");
	histHandler.calculate_cdf();

	// loop over full volume
	printf("[clahe3d] running histogram equilization for each voxel\n");
	histHandler.equalize();

	return;
}
