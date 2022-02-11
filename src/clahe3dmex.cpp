#include <fstream>
#include "interpGrid.h"
#include "histeq.h"
#include <mex.h>

using namespace std;

// clahe3dmex(interpVol, subVolSize, spacingSubVols, clipLimit, binSize);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	
	float* inputVol = (float *) mxGetData(prhs[0]); // input volume
	const uint64_t nZ = mxGetDimensions(prhs[0])[0]; // z dimension
	const uint64_t nX = mxGetDimensions(prhs[0])[1]; // x dimension
	const uint64_t nY = mxGetDimensions(prhs[0])[2]; // y dimension
	uint64_t * subVolSize = (uint64_t*) mxGetData(prhs[1]); // size of bins [z, x, y]
	uint64_t * subVolSpacing = (uint64_t*) mxGetData(prhs[2]); // size of bins [z, x, y]
	float* clipLimit = (float *) mxGetData(prhs[3]);
	uint64_t* binSize = (uint64_t*) mxGetData(prhs[4]);

	uint64_t volumeSize[3] = {nZ, nX, nY};
	

	printf("[clahe3d] initializing histogram handler\n");
	histeq histHandler;
	histHandler.setNBins(binSize[0]);
	histHandler.setNoiseLevel(clipLimit[0]);
	histHandler.setVolSize(volumeSize);
	histHandler.setSizeSubVols(subVolSize);
	histHandler.setSpacingSubVols(subVolSpacing);
	histHandler.setData(inputVol);
	
	printf("[clahe3d] calculating historgrams for subvolumes\n");
	histHandler.calculate();

	// loop over full volume
	printf("[clahe3d] running histogram equilization for each voxel\n");
	histHandler.equalize();

	return;
}
