#include "vector3.h"
#include "genfilt.h"
#include "cudaTools.cuh"

// struct containing all the settings for our mean filter
#ifndef MEANFILTSETT_H
#define MEANFILTSETT_H

struct meanfiltsett
{
	int kernelSize[3] = {3, 3, 3};
	bool flagGpu = 0;
};

#endif

// actual meanfilter class
#ifndef MEANFILT_H
#define MEANFILT_H

class meanfilt : 
#if USE_CUDA
	public cudaTools,
#endif
public genfilt
{
private:
	float* meanKernel;
	bool isKernelAlloc = 0;

public:
	meanfilt();
	~meanfilt();
	void run(); // runs the actual procesure
	#if USE_CUDA
	void run_gpu();
	#endif

};

#endif