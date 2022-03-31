#include "medianfilt.h"
#if USE_CUDA
	#include "medianfilt_kernel.cu"
#endif

// class constructor
medianfilt::medianfilt()
{

}

// runs the median filter over a specified range
void medianfilt::run_range(const int iRange)
{
	// temprary array used for sorting
	vector<float> sortArray(nKernel);
	vector<float> localArray(nKernel);

	for (auto ix = 0; ix < dataSize.x; ix++)
	{
		for (auto iy = 0; iy < dataSize.y; iy++)
		{
			// for first element along z we fill the entire array
			for (int zrel = 0; zrel < kernelSize.z; zrel++)
			{
				const int zAbs = zrel + zStart[iRange];
				for (int yrel = 0; yrel < kernelSize.y; yrel++)
				{
					const int yAbs = iy + yrel;
					const int idxPadd = ix + paddedSize.x * (yAbs + paddedSize.y * zAbs);
					const int idxKernel = kernelSize.x * (yrel + kernelSize.y * zrel);
					memcpy(&localArray[idxKernel], &dataPadded[idxPadd], kernelSize.x * sizeof(float));
				}
			}

			sortArray = localArray;
			std::nth_element(sortArray.begin(), sortArray.begin() + nKernel / 2, sortArray.end());
			const int idxOut1 = ix + dataSize.x * (iy + dataSize.y * zStart[iRange]);
			dataOutput[idxOut1] = sortArray[centerIdx];

			// now we start overwriting planes of memory
			int counter = 0;
			for (auto zAbs = zStart[iRange] + 1; zAbs <= zStop[iRange]; zAbs++)
			{
				// current plane we overwrite
				const int zkernel = counter % kernelSize.x;

				for (int yrel = 0; yrel < kernelSize.y; yrel++)
				{
					const int yAbs = iy + yrel;
					// start index in padded array for copy operation
					const int idxPadd = ix + paddedSize.x * (yAbs + paddedSize.y * zAbs);
					// start index in kernel array for copy operation
					const int idxKernel = kernelSize.x * (yrel + kernelSize.y * zkernel);
					memcpy(&localArray[idxKernel], &dataPadded[idxPadd], kernelSize.x * sizeof(float));
				}

				sortArray = localArray;
				std::nth_element(sortArray.begin(), sortArray.begin() + nKernel / 2, sortArray.end());
				// sort(sortArray.begin(), sortArray.end());
				// const int medianIdx = (kernelSize.x * kernelSize.y * nKernel - 1) / 2;



				const int idxOut2 = ix + dataSize.x * (iy + dataSize.y * zAbs);
				dataOutput[idxOut2] = sortArray[centerIdx];
				counter++;
			}
		}
	}

	return;
}

// performs median filtering on the CPU
void medianfilt::run()
{
	padd();
	alloc_output();

	nKernel = kernelSize.x * kernelSize.y * kernelSize.z;
	centerIdx = (nKernel - 1) / 2;
	sizeKernel = nKernel * sizeof(float);

	const auto tStart = std::chrono::high_resolution_clock::now();
	
	// define the range our threads need to calculate
	const int zRange = dataSize.z / nThreads;
	zStart.clear(); zStop.clear();
	for(std::size_t iThread = 0; iThread < nThreads; iThread++)
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
	for (std::size_t iThread = 0; iThread < nThreads; iThread++)
	{
		std::thread currThread(&medianfilt::run_range, this, iThread); 
		runners.push_back(std::move(currThread));
	}

	// collect all threads
	for (std::size_t iThread = 0; iThread < nThreads; iThread++)
	{
		runners[iThread].join();
	}
	
	const auto tStop = std::chrono::high_resolution_clock::now();
	const auto tDuration = std::chrono::duration_cast<std::chrono::milliseconds>(tStop- tStart);
	tExec = tDuration.count();

	return;
}


#if USE_CUDA

void medianfilt::run_gpu()
{
	padd();
	alloc_output();

	const auto tStart = std::chrono::high_resolution_clock::now();
	
	medianfilt_args args;
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		args.volSize[iDim] = (unsigned int)  dataSize[iDim]; 
		args.volSizePadded[iDim] = (unsigned int) paddedSize[iDim];
		args.kernelSize[iDim] = (unsigned int) kernelSize[iDim];
	}
	args.nKernel = get_nKernel();

	// allocate memory for input
	cudaError_t err = cudaMalloc((void**) &args.inputData, get_nPadded() * sizeof(float));
	checkCudaErr(err, "Could not allocate memory for input data");

	// copy memory over
	err = cudaMemcpy(args.inputData, dataPadded, get_nPadded() * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErr(err, "Could not copy memory to device");

	// allocate memory for output
	float* outData_dev;
	err = cudaMalloc((void**) &outData_dev, get_nData() * sizeof(float));
	checkCudaErr(err, "Could not allocate memory for output data");


	cudaDeviceProp props = get_devProps(0);
	const int sharedMemAvailable = props.sharedMemPerBlock;
	if (sharedMemAvailable < (args.nKernel * sizeof(float)))
	{
		printf("The kernel must fit into shared memory at least for one thread\n");
		throw "InvalidValue";
	}

	const int maxBlockSize = sharedMemAvailable / (args.nKernel * sizeof(float));
	const float nGridAccurate = powf((float) maxBlockSize, 0.33333f);
	
	printf("Maximum kernel size: %f\n", nGridAccurate);
	const dim3 blockSize(floor(nGridAccurate), floor(nGridAccurate), floor(nGridAccurate));

	const std::size_t bytesShared = blockSize.x * blockSize.y * blockSize.z * args.nKernel * sizeof(float);
	if (bytesShared > props.sharedMemPerBlock)
	{
		printf("Shared memory requested: %lu bytes\n", bytesShared);
		printf("Available shared memory: %d bytes\n", props.sharedMemPerBlock);
		throw "InvalidConfig";
	}

	const dim3 gridSize(
		(dataSize.x + blockSize.x - 1) / blockSize.x,
		(dataSize.y + blockSize.y - 1) / blockSize.y,
		(dataSize.z + blockSize.z - 1) / blockSize.z);

	medianfilt_cuda<<< gridSize, blockSize, bytesShared>>>(outData_dev, args);

	cudaDeviceSynchronize();

	err = cudaGetLastError();
	checkCudaErr(err, "Kernel crashed");

	// copy memory back from our precious GPU
	err = cudaMemcpy(dataOutput, outData_dev, get_nData() * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErr(err, "Could not copy data back from GPU");

	const auto tStop = std::chrono::high_resolution_clock::now();
	const auto tDuration = std::chrono::duration_cast<std::chrono::milliseconds>(tStop- tStart);
	tExec = tDuration.count();

	cudaFree(outData_dev);
	cudaFree(args.inputData);

	return;
}

#endif

// // creates a padded version of the input volume, order for median: y, z, x
// void medianfilt::padd()
// {
// 	// printf("Padding of median filter is running\n");
// 	paddedSize = get_paddedSize();
// 	alloc_padded();
// 	for (std::size_t iz = 0; iz < paddedSize.z; iz++)
// 	{
// 		const bool isZRange = ((iz >= range.z) && (iz <= (paddedSize.z - range.z - 1)));
// 		for (std::size_t iy = 0; iy < paddedSize.y; iy++)
// 		{
// 			const bool isYRange = ((iy >= range.y) && (iy <= (paddedSize.y - range.y - 1)));
// 			for (std::size_t ix = 0; ix < paddedSize.x; ix++)
// 			{
// 				const bool isXRange = ((ix >= range.x) && (ix <= (paddedSize.x - range.x - 1)));
// 				// if we are in valid volume, set to value, otherwise padd to 0 for now
// 				const std::size_t idxPad = iy + paddedSize.x * (iy + paddedSize.y * iz);
// 				if (isZRange && isXRange && isYRange)
// 				{
// 					const std::size_t idxInput = (ix - range.x) + dataSize.x * ((iy - range.y) + dataSize.y * (iz - range.z));
// 					// (iz - range.z)
// 					dataPadded[idxPad] = dataInput[idxInput];
// 				} 
// 				else // set 0 for now, later with symmetries etc.
// 				{
// 					dataPadded[idxPad] = 0;
// 				}
// 			}
// 		}
// 	}
// 	return;
// }