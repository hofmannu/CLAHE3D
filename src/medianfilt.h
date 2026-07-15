/*
	a class to run a medianfilter over a threedimensional volume
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 14.03.2022
*/

#include "genfilt.h"
#include "vector3.h"
#include <cstring>
#include <vector>
#include <algorithm>
#include <thread>

#if USE_CUDA
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_runtime_api.h>
	#include "cudaTools.cuh"
#endif

#ifndef MEDIANFILTSETT_H
#define MEDIANFILTSETT_H

struct medianfiltsett
{
	int kernelSize[3] = {3, 3, 3};
	bool flagGpu = 0;
};

#endif


#ifndef MEDIANFILT_H
#define MEDIANFILT_H

class medianfilt : public genfilt
{
private:
	// note: zStart / zStop are inherited (as vector<std::size_t>) from genfilt;
	// re-declaring them here as vector<int> previously shadowed the base members
	// and turned an underflowed SIZE_MAX range bound into int(-1).
	int nKernel; // number of elements in kernel
	int centerIdx; // center index of linearized kernel
	int sizeKernel; // size of kernel in bytes
	
	// void padd(); // function to apply padding to dataset
	void run_range(const int iRange); // run for a certain z range (multithread)
public:
	medianfilt(); // class constructor

	void run();
	void run_gpu();
};

#endif