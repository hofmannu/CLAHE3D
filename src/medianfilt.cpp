#include "medianfilt.h"

medianfilt::medianfilt()
{
	nThreads = std::thread::hardware_concurrency();
	// printf("Found a total of %d processing units\n", nThreads);
}

// runs the median filter over a specified range
void medianfilt::run_range(const int iRange)
{
	// temprary array used for sorting
	vector<float> sortArray(nKernel);
	float* localArray = new float [nKernel];

	for (auto iz = zStart[iRange]; iz < zStop[iRange]; iz++)
	{
		for (auto iy = 0; iy < dataSize.y; iy++)
		{
			// for first element along x we fill the entire array
			for (int xrel = 0; xrel < kernelSize.x; xrel++)
			{
				const int xAbs = xrel;
				for (int zrel = 0; zrel < kernelSize.z; zrel++)
				{
					const int zAbs = iz + zrel;

					const int idxPadd = iy + paddedSize.y * (zAbs + paddedSize.z * xAbs);
					const int idxKernel = kernelSize.y * (zrel + kernelSize.z * xrel);
					memcpy(&localArray[idxKernel], &dataPadded[idxPadd], kernelSize.y * sizeof(float));
					
				}
			}

			memcpy(sortArray.data(), localArray, sizeKernel);
			std::nth_element(sortArray.begin(), sortArray.begin() + nKernel / 2, sortArray.end());
			const int idxOut1 = dataSize.x * (iy + dataSize.y * iz);
			dataOutput[idxOut1] = sortArray[nKernel / 2];

			// now we start overwriting planes of memory
			for (auto ix = 1; ix < dataSize.x; ix++)
			{
				// current plane we overwrite
				const int xkernel = (ix - 1) % kernelSize.x;
				const int xAbs = ix + kernelSize.x - 1;

				for (int zrel = 0; zrel < kernelSize.z; zrel++)
				{
					const int zAbs = iz + zrel;
					// start index in padded array for copy operation
					const int idxPadd = iy + paddedSize.y * (zAbs + paddedSize.z * xAbs);
					// start index in kernel array for copy operation
					const int idxKernel = kernelSize.y * (zrel + kernelSize.z * xkernel);
					memcpy(&localArray[idxKernel], &dataPadded[idxPadd], kernelSize.y * sizeof(float));
				}

				memcpy(sortArray.data(), localArray, sizeKernel);
				std::nth_element(sortArray.begin(), sortArray.begin() + nKernel / 2, sortArray.end());

				const int idxOut2 = ix + dataSize.x * (iy + dataSize.y * iz);
				dataOutput[idxOut2] = sortArray[nKernel / 2];
			}
		}
	}
	delete[] localArray;

	return;
}

// performs median filtering
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
		std::thread currThread(&medianfilt::run_range, this, iThread); 
		runners.push_back(std::move(currThread));
	}

	// collect all threads
	for (int iThread = 0; iThread < nThreads; iThread++)
		runners[iThread].join();
	
	const auto tStop = std::chrono::high_resolution_clock::now();
	const auto tDuration = std::chrono::duration_cast<std::chrono::milliseconds>(tStop- tStart);
	tExec = tDuration.count();

	return;
}

// creates a padded version of the input volume, order for median: y, z, x
void medianfilt::padd()
{
	// printf("Padding of median filter is running\n");
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
				const int idxPad = iy + paddedSize.y * (iz + paddedSize.z * ix);
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