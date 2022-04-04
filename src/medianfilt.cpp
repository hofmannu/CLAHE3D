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
	// vector<float> sortArray(nKernel);
	vector<float> localArray(nKernel);

	for (std::size_t zOut = zStart[iRange]; zOut <= zStop[iRange]; zOut++)
	{
		for (std::size_t yOut = 0; yOut < dataSize.y; yOut++)
		{
			for (std::size_t xOut = 0; xOut < dataSize.x; xOut++)
			{
				// fill local array
				for (std::size_t zRel = 0; zRel < kernelSize.z; zRel++)
				{
					const std::size_t zAbs = zOut + zRel;
					for (std::size_t yRel = 0; yRel < kernelSize.y; yRel++)
					{
						const std::size_t yAbs = yOut + yRel;
						for (std::size_t xRel = 0; xRel < kernelSize.x; xRel++)
						{
							const std::size_t xAbs = xOut + xRel;
							const std::size_t idxGlobal = xAbs + paddedSize.x * (yAbs + paddedSize.y * zAbs);
							const std::size_t idxKernel = xRel + kernelSize.x * (yRel + kernelSize.y * zRel);
							localArray[idxKernel] = dataPadded[idxGlobal];
						}
					}
				}

				// sort
				std::nth_element(localArray.begin(), localArray.begin() + centerIdx, localArray.end());
				const std::size_t idxOut = xOut + dataSize.x * (yOut + dataSize.y * zOut);
				dataOutput[idxOut] = localArray[centerIdx];
			}
		}
	}

	// for (std::size_t ix = 0; ix < dataSize.x; ix++)
	// {
	// 	for (std::size_t iy = 0; iy < dataSize.y; iy++)
	// 	{
	// 		// for first element along z we fill the entire array
	// 		for (std::size_t zrel = 0; zrel < kernelSize.z; zrel++)
	// 		{
	// 			const std::size_t zAbs = zrel + zStart[iRange];
	// 			for (std::size_t yrel = 0; yrel < kernelSize.y; yrel++)
	// 			{
	// 				const std::size_t yAbs = iy + yrel;
	// 				for (std::size_t xrel = 0; xrel < kernelSize.x; xrel++)
	// 				{
	// 					const std::size_t xAbs = ix + xrel;
	// 					const std::size_t idxPadded = xAbs + paddedSize.x * (yAbs + paddedSize.y * zAbs);
	// 					const std::size_t idxKernel = xrel + kernelSize.x * (yrel + kernelSize.y * zrel);
	// 					localArray[idxKernel] = dataPadded[idxPadded];
	// 				}
	// 				// memcpy(&localArray[idxKernel], &dataPadded[idxPadd], kernelSize.x * sizeof(float));
	// 			}
	// 		}

	// 		sortArray = localArray;
	// 		std::nth_element(sortArray.begin(), sortArray.begin() + centerIdx, sortArray.end());
	// 		const std::size_t idxOut1 = ix + dataSize.x * (iy + dataSize.y * zStart[iRange]);
	// 		dataOutput[idxOut1] = sortArray[centerIdx];

	// 		// now we start overwriting planes of memory
	// 		std::size_t counter = 0;
	// 		for (std::size_t zAbs = zStart[iRange] + 1; zAbs <= zStop[iRange]; zAbs++)
	// 		{
	// 			// z index of next plane we want to overwrite
	// 			const std::size_t zkernel = counter % kernelSize.x;

	// 			for (std::size_t yrel = 0; yrel < kernelSize.y; yrel++)
	// 			{
	// 				const std::size_t yAbs = iy + yrel;
	// 				for (std::size_t xrel = 0; xrel < kernelSize.x; xrel++)
	// 				{
	// 					const std::size_t xAbs = ix + xrel;
	// 					const std::size_t idxPadded = xAbs + paddedSize.x * (yAbs + paddedSize.y * zAbs);
	// 					const std::size_t idxKernel = xrel + kernelSize.x * (yrel + kernelSize.y * zkernel);
	// 					localArray[idxKernel] = dataPadded[idxPadded];
	// 				}

	// 				// start index in padded array for copy operation
	// 				// const std::size_t idxPadd = ix + paddedSize.x * (yAbs + paddedSize.y * zAbs);
					
	// 				// start index in kernel array for copy operation
	// 				// const std::size_t idxKernel = 0 + kernelSize.x * (yrel + kernelSize.y * zkernel);
					
	// 				// copy a full line of memory
	// 				// memcpy(&localArray[idxKernel], &dataPadded[idxPadd], kernelSize.x * sizeof(float));
	// 			}


	// 			sortArray = localArray;
	// 			// sortArray.assign(localArray.begin(), localArray.end());

	// 			if ((zAbs == 40) && (ix == 40) && (iy == 40))
	// 			{
	// 				for (std::size_t idx = 0; idx < nKernel; idx++)
	// 				{
	// 					if ((idx % 25) == 0)
	// 						printf("\n");
	// 					printf("Index: %lu, Value: %.2f\n", idx, sortArray[idx]);
	// 				}
	// 			}

	// 			// note: we do not need to sort the entire array and nth element can be used to simply
	// 			// get the element at that certain position in the array
	// 			std::nth_element(sortArray.begin(), sortArray.begin() + centerIdx, sortArray.end());
				
	// 			const std::size_t idxOut2 = ix + dataSize.x * (iy + dataSize.y * zAbs);
	// 			dataOutput[idxOut2] = sortArray[centerIdx];
	// 			counter++;
	// 		}
	// 	}
	// }

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
	const std::size_t zRange = dataSize.z / nThreads;
	zStart.clear(); 
	zStop.clear();
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
		printf("Starting z: %lu, stopping z: %lu\n", zStart[iThread], zStop[iThread]);
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
