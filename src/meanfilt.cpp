#include "meanfilt.h"

meanfilt::meanfilt()
{
	set_kernelSize({5, 5, 5});
}

void meanfilt::run()
{
	const float scale = 1.0f / static_cast<float>(get_nKernel());
	meanKernel.resize(get_nKernel());
	for (int iKernel = 0; iKernel < get_nKernel(); iKernel++)
	{
		meanKernel[iKernel] = scale;
	}

	set_kernel(meanKernel.data());
	conv();
}

#if USE_CUDA
void meanfilt::run_gpu()
{

	const float scale = 1 / static_cast<float>(get_nKernel());
	meanKernel.resize(get_nKernel());
	for (int iKernel = 0; iKernel < get_nKernel(); iKernel++)
	{
		meanKernel[iKernel] = scale;
	}

	set_kernel(meanKernel.data());
	conv_gpu();
}

#endif