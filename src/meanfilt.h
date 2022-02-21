#include "vector3.h"
#include "genfilt.h"

#ifndef MEANFILT_H
#define MEANFILT_H

class meanfilt : public genfilt
{
private:
	float* meanKernel;
	bool isKernelAlloc = 0;

public:
	meanfilt();
	~meanfilt();
	void run();

};

#endif