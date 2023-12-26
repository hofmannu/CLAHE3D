#include "vector3.h"
#include "genfilt.h"
#include <cmath>
#include <vector>

#ifndef GAUSSFILTSETT_H
#define GAUSSFILTSETT_H

struct gaussfiltsett
{
	float sigma = 1.0f;
	int kernelSize[3] = {5, 5, 5};
	bool flagGpu = false;
};

#endif

#ifndef GAUSSFILT_H
#define GAUSSFILT_H

class gaussfilt : public genfilt 
{
public:
	gaussfilt();
	void run();
	#if USE_CUDA
	void run_gpu();
	#endif

	[[nodiscard]] float* get_psigma() {return &sigma;};
	void set_sigma(const float _sigma);
private:
	std::vector<float> gaussKernel;
	float sigma = 1.5f;
};

#endif