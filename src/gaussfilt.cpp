#include "gaussfilt.h"

gaussfilt::gaussfilt()
{
	set_kernelSize({9, 9, 9});
}

gaussfilt::~gaussfilt()
{
	if (isKernelAlloc)
		delete[] gaussKernel;
}

void gaussfilt::run()
{

	gaussKernel = new float[get_nKernel()];
	const vector3<std::size_t> kernelSize = get_kernelSize();
	const vector3<std::size_t> kernelRange = (kernelSize - 1) / 2;
	for (std::size_t iz = 0; iz < kernelSize.z; iz++)
	{
		const float dz = (float) iz - (float) kernelRange.z;
		for (std::size_t iy = 0; iy < kernelSize.y; iy++)
		{
			const float dy = (float) iy - (float) kernelRange.y;
			for (std::size_t ix = 0; ix < kernelSize.x; ix++)
			{
				const float dx = (float) ix - (float) kernelRange.x;
				const float dr = pow(dx * dx + dy * dy + dz * dz, 0.5f);
				const float currVal = expf(-1.0f / 2.0f / (dr * dr) / (sigma * sigma))
					/ (sigma * powf(2.0f * M_PI, 0.5f));
				const std::size_t idxLin = ix + kernelSize.x * (iy + kernelSize.y * iz);
				gaussKernel[idxLin] = currVal;
			}
		}
	}

	// normalize kernel to have a total value of 1
	float kernelSum = 0;
	for (std::size_t iElem = 0 ; iElem < get_nKernel(); iElem++)
		kernelSum += gaussKernel[iElem];
	
	for (std::size_t iElem = 0 ; iElem < get_nKernel(); iElem++)
		gaussKernel[iElem] = gaussKernel[iElem] / kernelSum;

	set_kernel(gaussKernel);
	conv();
	return;
}

// define the standard deviation
void gaussfilt::set_sigma(const float _sigma)
{
	sigma = _sigma;
	return;
}