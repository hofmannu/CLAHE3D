#include "vector3.h"
#include "genfilt.h"
#include <cmath>

#ifndef GAUSSFILTSETT_H
#define GAUSSFILTSETT_H

struct gaussfiltsett
{
	float sigma = 1;
	int kernelSize[3] = {5, 5, 5};
	bool flagGpu = 0;
};

#endif

#ifndef GAUSSFILT_H
#define GAUSSFILT_H

class gaussfilt : public genfilt 
{
private:
	float* gaussKernel;
	bool isKernelAlloc = 0;
	float sigma = 1.5f;
public:
	gaussfilt();
	~gaussfilt();
	void run();
	#if USE_CUDA
	void run_gpu();
	#endif

	float* get_psigma() {return &sigma;};
	void set_sigma(const float _sigma);
};

#endif