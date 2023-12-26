#include "vector3.h"
#include "genfilt.h"
#include <vector>

#if USE_CUDA
#include "cudaTools.cuh"
#endif

// struct containing all the settings for our mean filter
#ifndef MEANFILTSETT_H
#define MEANFILTSETT_H

struct meanfiltsett
{
	int kernelSize[3] = {3, 3, 3};
	bool flagGpu = false;
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
public:
	meanfilt();
	void run(); // runs the actual procesure
#if USE_CUDA
	void run_gpu();
#endif
private:
	std::vector<float> meanKernel;



};

#endif