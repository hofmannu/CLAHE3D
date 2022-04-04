#include "meanfilt.h"

meanfilt::meanfilt()
{
	set_kernelSize({5, 5, 5});
}

meanfilt::~meanfilt()
{
	if (isKernelAlloc)
		delete[] meanKernel;
}

void meanfilt::run()
{

	const float scale = 1 / ((float) get_nKernel());

	// allocate memory for kernel
	if (isKernelAlloc)
		delete[] meanKernel;

	meanKernel = new float[get_nKernel()];
	for (int iKernel = 0; iKernel < get_nKernel(); iKernel++)
	{
		meanKernel[iKernel] = scale;
	}

	set_kernel(meanKernel);
	conv();
	return;
}

#if USE_CUDA
void meanfilt::run_gpu()
{

	const float scale = 1 / ((float) get_nKernel());

	// allocate memory for kernel
	if (isKernelAlloc)
		delete[] meanKernel;

	meanKernel = new float[get_nKernel()];
	for (int iKernel = 0; iKernel < get_nKernel(); iKernel++)
	{
		meanKernel[iKernel] = scale;
	}

	set_kernel(meanKernel);
	conv_gpu();
	return;
}

#endif