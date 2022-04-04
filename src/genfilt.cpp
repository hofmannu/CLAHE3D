#include "genfilt.h"

#if USE_CUDA
	#include "genfilt_kernel.cu"
#endif
// class constructor and destructor
genfilt::genfilt()
{
	nThreads = std::thread::hardware_concurrency();
	// printf("Running on %d threads\n", nThreads);
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
	for (std::size_t iz = 0; iz < paddedSize.z; iz++)
	{
		const bool isZRange = ((iz >= range.z) && (iz <= (paddedSize.z - range.z - 1)));
		for (std::size_t iy = 0; iy < paddedSize.y; iy++)
		{
			const bool isYRange = ((iy >= range.y) && (iy <= (paddedSize.y - range.y - 1)));
			for (std::size_t ix = 0; ix < paddedSize.x; ix++)
			{
				const bool isXRange = ((ix >= range.x) && (ix <= (paddedSize.x - range.x - 1)));
				// if we are in valid volume, set to value, otherwise padd to 0 for now
				const std::size_t idxPad = ix + paddedSize.x * (iy + paddedSize.y * iz);
				if (isZRange && isXRange && isYRange)
				{
					const std::size_t idxInput = (ix - range.x) + dataSize.x * ((iy - range.y) + dataSize.y * (iz - range.z));
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
void genfilt::conv_range(const std::size_t iRange)
{
	for (std::size_t iz = zStart[iRange]; iz <= zStop[iRange]; iz++)
	{
		for (std::size_t iy = 0; iy < dataSize.y; iy++)
		{
			for (std::size_t ix = 0; ix < dataSize.x; ix++)
			{
				// for each output element we do the sum with the local kernel
				float tempVal = 0;
				for (std::size_t zrel = 0; zrel < kernelSize.z; zrel++)
				{
					const std::size_t zAbs = iz + zrel;
					for (std::size_t yrel = 0; yrel < kernelSize.y; yrel++)
					{
						const std::size_t yAbs = iy + yrel;
						for (std::size_t xrel = 0; xrel < kernelSize.x; xrel++)
						{
							
							const std::size_t xAbs = ix + xrel; // get current index in padding

							// index in padded volume
							const std::size_t idxPadd = xAbs + paddedSize.x * (yAbs + paddedSize.y * zAbs);

							const std::size_t idxKernel = xrel + kernelSize.x * (yrel + kernelSize.y * zrel);

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

#if USE_CUDA
// runs the kernel convolution on the GPU
void genfilt::conv_gpu()
{
	padd();
	alloc_output();

	const auto tStart = std::chrono::high_resolution_clock::now();

	genfilt_args args;
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		args.volSize[iDim] = (unsigned int)  dataSize[iDim]; 
		args.volSizePadded[iDim] = (unsigned int) paddedSize[iDim];
		args.kernelSize[iDim] = (unsigned int) kernelSize[iDim];
	}
	args.nKernel = get_nKernel();

	// memory allocation
	const std::size_t memRequired = (get_nPadded() + get_nKernel() + get_nData()) * sizeof(float);
	cudaDeviceProp props = get_devProps(0);
	if (props.totalGlobalMem < memRequired)
	{
		printf("There is insufficient space on the device to allocate stuff\n");
		throw "InvalidValue";
	}

	cudaError_t err = cudaMalloc((void**) &args.inputData, get_nPadded() * sizeof(float));
	checkCudaErr(err, "COuld not allocate memory for input dataset on GPU");

	err = cudaMalloc((void**) &args.kernel, get_nKernel() * sizeof(float));
	checkCudaErr(err, "Could not allocate memory for kernel");

	float* outData_dev;
	err = cudaMalloc((void**) &outData_dev, get_nData() * sizeof(float));
	checkCudaErr(err, "Could not allocate memory for output array on device");

	// copy memory to device
	err = cudaMemcpy(args.inputData, dataPadded, get_nPadded() * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErr(err, "Could not copy input dataset to device");

	err = cudaMemcpy(args.kernel, kernel, get_nKernel() * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErr(err, "Could not copy kernel to device");
	

	const dim3 blockSize(8, 8, 8);

	args.localSize[0] = blockSize.x + args.kernelSize[0] - 1;
	args.localSize[1] = blockSize.y + args.kernelSize[1] - 1;
	args.localSize[2] = blockSize.z + args.kernelSize[2] - 1;
	
	const std::size_t sizeSlice = args.localSize[0] * args.localSize[1] * sizeof(float);
	const std::size_t nBytesShared = get_nKernel() * sizeof(float) + sizeSlice;
	if (nBytesShared > props.sharedMemPerBlock)
	{
		printf("Amount of shared memory required for this: %lu bytes\n", nBytesShared);
		printf("Even smallest block size won't fit on shared\n");
		throw "InvalidValue";
	} 


	// prepare launch parameters
	const dim3 gridSize(
		(dataSize.x + blockSize.x - 1) / blockSize.x,
		(dataSize.y + blockSize.y - 1) / blockSize.y,
		(dataSize.z + blockSize.z - 1) / blockSize.z);

	// launch kernel
	genfilt_cuda<<<gridSize, blockSize, nBytesShared>>>(outData_dev, args);
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	checkCudaErr(err, "Kernel crashed");

	// copy memory back from device
	err = cudaMemcpy(dataOutput, outData_dev, get_nData() * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErr(err, "Could not copy data back from GPU");

	const auto tStop = std::chrono::high_resolution_clock::now();
	const auto tDuration = std::chrono::duration_cast<std::chrono::milliseconds>(tStop- tStart);
	tExec = tDuration.count();

	cudaFree(args.inputData);
	cudaFree(args.kernel);
	cudaFree(outData_dev);
	return;

}
#endif

// performs the convolution of the dataset with the kernel on the CPU
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
	for (uint8_t iThread = 0; iThread < nThreads; iThread++)
	{
		std::thread currThread(&genfilt::conv_range, this, iThread); 
		runners.push_back(std::move(currThread));
	}

	// collect all threads
	for (uint8_t iThread = 0; iThread < nThreads; iThread++)
		runners[iThread].join();

	const auto tStop = std::chrono::high_resolution_clock::now();
	const auto tDuration = std::chrono::duration_cast<std::chrono::milliseconds>(tStop - tStart);
	tExec = tDuration.count();
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

// define the size of the input dataset
void genfilt::set_dataSize(const vector3<std::size_t> _dataSize)
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

// define the size of the convolution kernel
void genfilt::set_kernelSize(const vector3<std::size_t> _kernelSize)
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
