#include "genfilt.h"

genfilt::~genfilt()
{
	if (isDataOutputAlloc)
		delete[] dataOutput;

	if (isDataPaddedAlloc)
		delete[] dataPadded;

	return;
}

void genfilt::alloc_output()
{
	if (isDataOutputAlloc)
		delete[] dataOutput;

	dataOutput = new float [get_nData()];
	isDataOutputAlloc = 1;
	return;
}

// allocate memory for padded data array
void genfilt::alloc_padded()
{
	if (isDataPaddedAlloc)
		delete[] dataPadded;

	dataPadded = new float [get_nPadded()];
	isDataPaddedAlloc = 1;
	return;
}

// creates a padded version of the input volume
void genfilt::padd()
{
	paddedSize = get_paddedSize();
	alloc_padded();
	for (int iz = 0; iz < paddedSize.z; iz++)
	{
		const bool isZRange = ((iz >= range.z) && (iz <= (paddedSize.z - range.z - 1)));
		for (int iy = 0; iy < paddedSize.y; iy++)
		{
			const bool isYRange = ((iy >= range.y) && (iy <= (paddedSize.y - range.y - 1)));
			for (int ix = 0; ix < paddedSize.x; ix++)
			{
				const bool isXRange = ((ix >= range.x) && (ix <= (paddedSize.x - range.x - 1)));
				// if we are in valid volume, set to value, otherwise padd to 0 for now
				const int idxPad = ix + paddedSize.x * (iy + paddedSize.y * iz);
				if (isZRange && isXRange && isYRange)
				{
					const int idxInput = (ix - range.x) + dataSize.x * ((iy - range.y) + dataSize.y * (iz - range.z));
					// (iz - range.z)
					dataPadded[idxPad] = dataInput[idxInput];
				} 
				else // set 0 for now, later with symmetries etc.
				{
					dataPadded[idxPad] = 0;
				}
			}
		}
	}
	return;
}

void genfilt::conv()
{	
	printf("Executing filter\n");
	printf(" - kernel size: %d x %d x %d\n");

	padd();
	alloc_output();
	for (int iz = 0; iz < dataSize.z; iz++)
	{
		for (int iy = 0; iy < dataSize.y; iy++)
		{
			for (int ix = 0; ix < dataSize.x; ix++)
			{
				// for each output element we do the sum with the local kernel
				float tempVal = 0;
				for (int zrel = -range.z; zrel <= range.z; zrel++)
				{
					for (int yrel = -range.y; yrel <= range.y; yrel++)
					{
						for (int xrel = -range.x; xrel <= range.x; xrel++)
						{
							// get current index in padding
							const int xAbs = ix + xrel;
							const int yAbs = iy + yrel;
							const int zAbs = iz + zrel;
							const int idxPadd = xAbs + paddedSize.x * (yAbs + paddedSize.y * zAbs);

							// get current index in kernel
							const int xKernel = xrel + range.x;
							const int yKernel = yrel + range.y;
							const int zKernel = zrel + range.z;
							const int idxKernel = xKernel + kernelSize.x * (yKernel + kernelSize.y * zKernel);

							tempVal += (dataPadded[idxPadd] * kernel[idxKernel]);
						}
					}
				}
				const int idxOut = ix + dataSize.x * (iy + dataSize.y * iz);
				dataOutput[idxOut] = tempVal;
			}
		}
	}
	return;
}

void genfilt::conv_gpu()
{

	return;
}

// set functions
void genfilt::set_dataInput(float* _dataInput)
{
	dataInput = _dataInput;
	return;
}

void genfilt::set_kernel(float* _kernel)
{
	kernel = _kernel;
	return;
}


void genfilt::set_dataSize(const vector3<int> _dataSize)
{
	#pragma unroll
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		if (_dataSize[iDim] < 1)
		{
			printf("Kernel size must be at least 1 in each dimension");
			throw "InvalidConfig";
		}
	}
	dataSize = _dataSize;
	return;
}

void genfilt::set_kernelSize(const vector3<int> _kernelSize)
{
	#pragma unroll
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		if (_kernelSize[iDim] < 1)
		{
			printf("Kernel size must be at least 1 in each dimension");
			throw "InvalidConfig";
		}

		if ((_kernelSize[iDim] % 2) == 0)
		{
			printf("Kernel must be uneven in each dimension");
			throw "InvalidConfig";
		}
	}
	kernelSize = _kernelSize;
	range = (kernelSize - 1) / 2;
	return;
}
