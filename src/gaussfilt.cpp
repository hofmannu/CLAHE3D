#include "gaussfilt.h"

gaussfilt::gaussfilt()
{
	set_kernelSize({9, 9, 9});
}

void gaussfilt::run()
{
	gaussKernel.resize(get_nKernel());
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
				// true Gaussian weight exp(-r^2 / (2 sigma^2)); the constant 1/(sigma*sqrt(2pi))
			// prefactor is omitted because the kernel is normalised to sum 1 below. The
			// previous form exp(-1/(2 r^2 sigma^2)) was not a Gaussian and made the centre
			// tap (r == 0) zero instead of the peak.
			const float currVal = expf(-(dr * dr) / (2.0f * sigma * sigma));
				const std::size_t idxLin = ix + kernelSize.x * (iy + kernelSize.y * iz);
				gaussKernel[idxLin] = currVal;
			}
		}
	}

	// normalize kernel to have a total value of 1
	float kernelSum = 0.0f;
	for (std::size_t iElem = 0 ; iElem < get_nKernel(); iElem++)
		kernelSum += gaussKernel[iElem];
	
	for (std::size_t iElem = 0 ; iElem < get_nKernel(); iElem++)
		gaussKernel[iElem] = gaussKernel[iElem] / kernelSum;

	set_kernel(gaussKernel.data());
	conv();
}

#if USE_CUDA
void gaussfilt::run_gpu()
{

	gaussKernel.resize(get_nKernel());
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
				// true Gaussian weight exp(-r^2 / (2 sigma^2)); the constant 1/(sigma*sqrt(2pi))
			// prefactor is omitted because the kernel is normalised to sum 1 below. The
			// previous form exp(-1/(2 r^2 sigma^2)) was not a Gaussian and made the centre
			// tap (r == 0) zero instead of the peak.
			const float currVal = expf(-(dr * dr) / (2.0f * sigma * sigma));
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

	set_kernel(gaussKernel.data());
	conv_gpu();
}
#endif


// define the standard deviation
void gaussfilt::set_sigma(const float _sigma)
{
	sigma = _sigma;
}