#include "genfilt.h"

// class constructor and destructor
genfilt::genfilt()
{
	nThreads = std::thread::hardware_concurrency();
	printf("Running on %d threads\n", nThreads);
}

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

// creates a padded version of the input volume  for now we simply pad with zeros
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

// convolution procedure for subrange of volume along z
void genfilt::conv_range(const int iRange)
{
	for (int iz = zStart[iRange]; iz <= zStop[iRange]; iz++)
	{
		for (int iy = 0; iy < dataSize.y; iy++)
		{
			for (int ix = 0; ix < dataSize.x; ix++)
			{
				// for each output element we do the sum with the local kernel
				float tempVal = 0;
				for (int zrel = 0; zrel < kernelSize.z; zrel++)
				{
					const int zAbs = iz + zrel;
					for (int yrel = 0; yrel < kernelSize.y; yrel++)
					{
						const int yAbs = iy + yrel;
						for (int xrel = 0; xrel < kernelSize.x; xrel++)
						{
							
							const int xAbs = ix + xrel; // get current index in padding

							// index in padded volume
							const int idxPadd = xAbs + paddedSize.x * (yAbs + paddedSize.y * zAbs);

							const int idxKernel = xrel + kernelSize.x * (yrel + kernelSize.y * zrel);

							tempVal = fmaf(
								dataPadded[idxPadd], kernel[idxKernel], tempVal);
						}
					}
				}
				const int idxOut = ix + dataSize.x * (iy + dataSize.y * iz);
				dataOutput[idxOut] = tempVal;
			}
		}
	}
}

void genfilt::conv()
{
	padd();
	alloc_output();
	const auto tStart = std::chrono::high_resolution_clock::now();

	const int zRange = dataSize.z / nThreads;
	zStart.clear(); zStop.clear();
	for(int iThread = 0; iThread < nThreads; iThread++)
	{
		zStart.push_back(iThread * zRange);
		if (iThread < (nThreads - 1))
		{
			zStop.push_back((iThread + 1) * zRange - 1);
		}
		else
		{
			zStop.push_back(dataSize.z - 1);
		}
	}

	// create a container for our multithread processing units
	std::vector<thread> runners;
	
	// launch all threads
	for (int iThread = 0; iThread < nThreads; iThread++)
	{
		std::thread currThread(&genfilt::conv_range, this, iThread); 
		runners.push_back(std::move(currThread));
	}

	// collect all threads
	for (int iThread = 0; iThread < nThreads; iThread++)
		runners[iThread].join();

	const auto tStop = std::chrono::high_resolution_clock::now();
	const auto tDuration = std::chrono::duration_cast<std::chrono::milliseconds>(tStop - tStart);
	tExec = tDuration.count();
	return;
}


#if USE_CUDA
// TODO: not implemented yet
void genfilt::conv_gpu()
{

	return;
}
#endif

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
