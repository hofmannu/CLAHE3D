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
	const vector3<int> kernelSize = get_kernelSize();
	const vector3<int> kernelRange = (kernelSize - 1) / 2;
	for (int iz = 0; iz < kernelSize.z; iz++)
	{
		const float dz = (float) iz - kernelRange.z;
		for (int iy = 0; iy < kernelSize.y; iy++)
		{
			const float dy = (float) iy - kernelRange.y;
			for (int ix = 0; ix < kernelSize.x; ix++)
			{
				const float dx = (float) ix - kernelRange.x;
				const float dr = pow(dx * dx + dy * dy + dz * dz, 0.5f);
				const float currVal = exp(-1.0f / 2.0f / (dr * dr) / (sigma * sigma))
					/ (sigma * pow(2.0f * M_PI, 0.5f));
				const int idxLin = ix + kernelSize.x * (iy + kernelSize.y * iz);
				gaussKernel[idxLin] = currVal;
			}
		}
	}

	// normalize kernel to have a total value of 1
	float kernelSum = 0;
	for (int iElem = 0 ; iElem < get_nKernel(); iElem++)
		kernelSum += gaussKernel[iElem];
	
	for (int iElem = 0 ; iElem < get_nKernel(); iElem++)
		gaussKernel[iElem] = gaussKernel[iElem] / kernelSum;

	set_kernel(gaussKernel);
	conv();
	return;
}