#include "vector3.h"
#include "genfilt.h"
#include <cmath>

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

	float* get_psigma() {return &sigma;};
};

#endif