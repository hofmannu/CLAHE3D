#include "histeq.h"
#if USE_CUDA
	#include "histeq_kernel.cu" // all declarations for cuda are externalized here
#endif

histeq::histeq()
{
	// get number of multiprocessing units
	nThreads = std::thread::hardware_concurrency();
}

histeq::~histeq()
{
	// free all memory once we finish this here
	if (isCdfAlloc)
		delete[] cdf;

	if (isMaxValBinAlloc)
	{
		delete[] minValBin;
		delete[] maxValBin;
	}

	if (isDataOutputAlloc)
	{
		delete[] dataOutput;
	}
}

#if USE_CUDA
// same as calculate but this time running on the GPU
void histeq::calculate_cdf_gpu()
{
	calculate_nsubvols();
	
	// define grid and block size
	const dim3 blockSize(4, 4, 4);
	const dim3 gridSize(
		(nSubVols[0] + blockSize.x - 1) / blockSize.x,
		(nSubVols[1] + blockSize.y - 1) / blockSize.y,
		(nSubVols[2] + blockSize.z - 1) / blockSize.z);

	// prepare input argument struct
	cdf_arguments inArgs;
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		inArgs.spacingSubVols[iDim] = spacingSubVols[iDim]; // size of each subvolume
		inArgs.nSubVols[iDim] = nSubVols[iDim]; // number of subvolumes
		inArgs.volSize[iDim] = volSize[iDim]; // overall size of data volume
		inArgs.range[iDim] = (int) (sizeSubVols[iDim] - 1) / 2; // range of each bin in each direction
		inArgs.sizeSubVols[iDim] = sizeSubVols[iDim];
		inArgs.origin[iDim] = origin[iDim];
	}
	inArgs.nBins = nBins;
	inArgs.noiseLevel = noiseLevel;

	float* dataMatrix_dev; // pointer to data matrix clone on GPU (read only)
	float* cdf_dev; // cumulative distribution function [iBin, izSub, ixSub, iySub]
	float* maxValBin_dev; // maximum value of current bin [izSub, ixSub, iySub]
	float* minValBin_dev; // minimum value of current bin [izSub, ixSub, iySub]
	const int nCdf = nBins * get_nSubVols();

	cudaError_t cErr;

	// allocate memory for main data array
	cErr = cudaMalloc((void**)&dataMatrix_dev, nElements * sizeof(float) );
	checkCudaErr(cErr, "Could not allocate memory for inputData on GPU");

	// allocate memory for cumulative distribution function
	cErr = cudaMalloc((void**)&cdf_dev, nCdf * sizeof(float));
	checkCudaErr(cErr, "Could not allocate memory for cdf on GPU");

	cErr = cudaMemcpyToSymbol(inArgsCdf, &inArgs, sizeof(cdf_arguments));
	checkCudaErr(cErr, "Could not copy symbol to GPU");

	// allocate memory for maximum values in bins
	cErr = cudaMalloc((void**)&maxValBin_dev, get_nSubVols() * sizeof(float));
	checkCudaErr(cErr, "Could not allocate memory for maxValBins on GPU");

	// allocate memory for minimum values in bins
	cErr = cudaMalloc((void**)&minValBin_dev, get_nSubVols() * sizeof(float));
	checkCudaErr(cErr, "Coult not allocate memory for miNValBins on GPU");

	// copy data matrix over
	cErr = cudaMemcpy(dataMatrix_dev, dataMatrix, nElements * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErr(cErr, "Could not copy data array to GPU");
	
	// here we start the execution of the first kernel (dist function)
	const auto startTimeCdf = std::chrono::high_resolution_clock::now();

	cdf_kernel<<< gridSize, blockSize>>>(
		cdf_dev, // pointer to cumulative distribution function 
		maxValBin_dev, // pointer to maximum values bin
		minValBin_dev, // pointer to minimum values bin
		dataMatrix_dev // data matrix
		);
	
	// wait for GPU calculation to finish before we copy things back
	cudaDeviceSynchronize();
	const auto stopTimeCdf = std::chrono::high_resolution_clock::now();
	const auto duractionCdf = std::chrono::duration_cast<std::chrono::milliseconds>
		(stopTimeCdf - startTimeCdf);
	printf("Time required for CDF kernels: %d ms\n", duractionCdf.count());

	// check if there was any problem during kernel execution
	cErr = cudaGetLastError();
	checkCudaErr(cErr, "Error during cdf-kernel execution");

	// allocate memory for transfer function
	if (isCdfAlloc)
		delete[] cdf;
	cdf = new float[nCdf];
	isCdfAlloc = 1;
	
	// copy transfer function back from device
	cErr = cudaMemcpy(cdf, cdf_dev, nCdf * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErr(cErr, "Problem while copying cdf back from device");

	// allocate memory for maximum and minimum values of our bins
	if (isMaxValBinAlloc)
	{
		delete[] maxValBin;
		delete[] minValBin;
	}
	maxValBin = new float[get_nSubVols()];
	minValBin = new float[get_nSubVols()];
	isMaxValBinAlloc = 1;

	// copy back minimum and maximum values of the bins from our GPU
	cErr = cudaMemcpy(maxValBin, maxValBin_dev, get_nSubVols() * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErr(cErr, "Could not copy back maximum values of our bins from device");

	cErr = cudaMemcpy(minValBin, minValBin_dev, get_nSubVols() * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErr(cErr, "Could not copy back minimum values of our bins from device");

	cudaFree(dataMatrix_dev);
	cudaFree(cdf_dev);
	cudaFree(maxValBin_dev);
	cudaFree(minValBin_dev);
	return;
}
#endif

void histeq::calculate_cdf_range(const std::size_t zStart, const std::size_t zStop)
{
	// allocate a little helper array
	const std::size_t nElementsLocal = sizeSubVols.x * sizeSubVols.y * sizeSubVols.z;
	float* localData = new float [nElementsLocal];
	
	// calculate histogram for each individual block
	for (std::size_t iZSub = zStart; iZSub <= zStop; iZSub++) // for each x subvolume
	{	
		for(std::size_t iYSub = 0; iYSub < nSubVols.y; iYSub++) // for each z subvolume
		{
			for(std::size_t iXSub = 0; iXSub < nSubVols.x; iXSub++) // for each y subvolume
			{
				const vector3<std::size_t> idxSub = {iXSub, iYSub, iZSub}; // index of current subvolume
				const vector3<std::size_t> idxStart = get_startIdxSubVol(idxSub);
				const vector3<std::size_t> idxEnd = get_stopIdxSubVol(idxSub);
				// printf("subvol ranges from %d, %d, %d to %d, %d, %d\n",
				// 	idxStart.x, idxStart.y, idxStart.z, idxEnd.x, idxEnd.y, idxEnd.z);
				calculate_sub_cdf(idxStart, idxEnd, idxSub, localData);
			}
		}
	}
	delete[] localData;

}

// cpu version of cummulative distribution function calculation
void histeq::calculate_cdf()
{
	const auto tStart = std::chrono::high_resolution_clock::now();
	calculate_nsubvols();
	
	// allocate memory for transfer function
	if (isCdfAlloc)
		delete[] cdf;
	cdf = new float[nBins * get_nSubVols()];
	isCdfAlloc = 1;

	if (isMaxValBinAlloc)
	{
		delete[] maxValBin;
		delete[] minValBin;
	}
	maxValBin = new float[get_nSubVols()];
	minValBin = new float[get_nSubVols()];
	isMaxValBinAlloc = 1;

	std::size_t activeWorkers = 0;
	vector<std::size_t> zStart;
	vector<std::size_t> zStop;

	workers.clear();
	if (nSubVols.z <= nThreads)
	{
		activeWorkers = nSubVols.z;
		for (std::size_t iWorker = 0; iWorker < activeWorkers; iWorker++)
		{
			zStart.push_back(iWorker);
			zStop.push_back(iWorker);
		}
	}
	else
	{
		activeWorkers = nThreads;
		std::size_t currPos = 0;
		for (std::size_t iWorker = 0; iWorker < activeWorkers; iWorker++)
		{
			zStart.push_back(currPos);
			currPos += (nSubVols.z / activeWorkers) - 1;
			if (iWorker < (nSubVols.z % activeWorkers))
				currPos++;

			zStop.push_back(currPos);
			currPos++;
		}
	}

	for (std::size_t iWorker = 0; iWorker < activeWorkers; iWorker++)
	{
		std::thread currThread(&histeq::calculate_cdf_range, this, zStart[iWorker], zStop[iWorker]);
		workers.push_back(std::move(currThread));
	}

	for (std::size_t iWorker = 0; iWorker < activeWorkers; iWorker++)
		workers[iWorker].join();

	const auto tStop = std::chrono::high_resolution_clock::now();
	const auto tDuration = std::chrono::duration_cast<std::chrono::milliseconds>(tStop- tStart);
	tCdf = tDuration.count();
	return;
}

// get cummulative distribution function for a certain bin 
void histeq::calculate_sub_cdf(
	const vector3<std::size_t>& startVec, 
	const vector3<std::size_t>& endVec, 
	const vector3<std::size_t>& iBin,
	float* localData) // bin index
{
	const std::size_t idxSubVol = iBin.x + nSubVols.x * (iBin.y + iBin.z * nSubVols.y);
	float* localCdf = &cdf[idxSubVol * nBins]; // histogram of subvolume, only temporarily requried

	// calculate number of elements for this local operation (can be smaller then array size)	
	const vector3<std::size_t> nElementsLocal = endVec - startVec + 1;
	const std::size_t nDataLocal = nElementsLocal.x * nElementsLocal.y * nElementsLocal.z;

	// variables holding min and max value
	const std::size_t firstElem = startVec.x + volSize.x * (startVec.y + volSize.y * startVec.z);
	float tempMax = dataMatrix[firstElem];
	float tempMin = dataMatrix[firstElem];

	// copy array to local and check for min and max
	std::size_t localIdx = 0;
	for (std::size_t iZ = startVec.z; iZ <= endVec.z; iZ++)
	{
		const std::size_t zOffset = iZ * volSize.x * volSize.y;
		for(std::size_t iY = startVec.y; iY <= endVec.y ; iY++)
		{
			const std::size_t yOffset = iY * volSize.x;
			for(std::size_t iX = startVec.x; iX <= endVec.x; iX++)
			{
				const float currVal = dataMatrix[iX + yOffset + zOffset];
				
				// copy to local data array
				localData[localIdx] = currVal;
				localIdx++;

				// check if new min or max
				if (currVal > tempMax)
				{
					tempMax = currVal;
				}
				if (currVal < tempMin)
				{
					tempMin = currVal;
				}
			}
		}
	}

	// only continue if we found any element which is above noise level
	if (tempMax > noiseLevel)
	{

		// reset bins to zero before summing them up
		for (std::size_t iBin = 0; iBin < nBins; iBin++)
			localCdf[iBin] = 0.0f;
		
		maxValBin[idxSubVol] = tempMax;
		tempMin = (tempMin < noiseLevel) ? noiseLevel : tempMin;
		minValBin[idxSubVol] = tempMin;

		// calculate size of each bin
		const float binRange = (tempMin == tempMax) ? 
			1.0f : (tempMax - tempMin) / ((float) nBins);

		// sort values into bins which are above clipLimit
		for (std::size_t iElem = 0; iElem < nDataLocal; iElem++)
		{
			const float currVal = localData[iElem];
			// only add to histogram if above clip limit
			if (currVal >= noiseLevel)
			{
				const std::size_t iBin = (currVal - tempMin) / binRange;
				// special case for maximum values in subvolume (they gonna end up
				// one index above)

				if (iBin >= nBins)
				{
					localCdf[nBins - 1] += 1.0f;
				}
				else
				{
					localCdf[iBin] += 1.0f;
				}
			}
		}

		// calculate cummulative sum and scale along y
		float cdfTemp = 0.0f;
		for (std::size_t iBin = 0; iBin < nBins; iBin++)
		{
			cdfTemp += localCdf[iBin];
			localCdf[iBin] = cdfTemp;
		}

		// now we scale cdf to max == 1 (first value is 0 anyway)
		const float zeroElem = localCdf[0];
		const float valRange = localCdf[nBins - 1] - zeroElem;
		const float rvalRange = 1.0f / valRange;
		for (std::size_t iBin = 0; iBin < nBins; iBin++)
		{
			localCdf[iBin] = (localCdf[iBin] - zeroElem) * rvalRange;
		}
		
	}
	else // if no value was above noise level --> linear cdf
	{
		maxValBin[idxSubVol] = noiseLevel;
		minValBin[idxSubVol] = noiseLevel;
		for (std::size_t iBin = 0; iBin < (nBins - 1); iBin++)
		{
			localCdf[iBin] = ((float) iBin) / ((float) nBins - 1);
		}
	}

	return;
}

// returns the inverted cummulative distribution function
template<typename T>
inline float histeq::get_icdf(const vector3<T>& iSubVol, const float currValue) const // value to extract
{
	return get_icdf(iSubVol.x, iSubVol.y, iSubVol.z, currValue);
}

// returns the inverted cummulative distribution function
template<typename T>
inline float histeq::get_icdf(const T ix, const T iy, const T iz, const float currValue) const // value to extract
{
	// linear index of current subvolume 
	const T subVolIdx = ix + nSubVols.x * (iy + nSubVols.y * iz);
	
	// if we are below noise level, directy return
	if (currValue <= minValBin[subVolIdx])
	{
		return 0.0f;
	}
	else
	{
		const T subVolOffset = nBins * subVolIdx;
		const float vInterp = (currValue - minValBin[subVolIdx]) / 
			(maxValBin[subVolIdx] - minValBin[subVolIdx]); // should now span 0 to 1 		
		
		// it can happen that the voxel value is higher then the max value detected
		// in the next volume. In this case we crop it to the maximum permittable value
		const T binOffset = (vInterp > 1.0f) ? 
			(nBins - 1 + subVolOffset)
			: fmaf(vInterp, (float) nBins - 1.0f, 0.5f) + subVolOffset;


		return cdf[binOffset];
	}
}

#if USE_CUDA
// prepare and launch the kernel responsible for histeq
void histeq::equalize_gpu()
{
	// define grid size (not optimized)
	const dim3 blockSize(32, 2, 2);
	const dim3 gridSize(
		(volSize[0] + blockSize.x - 1) / blockSize.x,
		(volSize[1] + blockSize.y - 1) / blockSize.y,
		(volSize[2] + blockSize.z - 1) / blockSize.z);

	// allocate memory on GPU
	float* dataMatrix_dev; // array for data matrix [iz, ix, iy]
	float* cdf_dev; // array for cummulative distribution function [iBin, iZS, iXS, iYS]
	float* minValBin_dev; // minimum value of each bin [iZS, iXS, iYS]
	float* maxValBin_dev; // maximum value of each bin [iZS, iXS, iYS]
	const int nCdf = nBins * get_nSubVols();

	// allocate memory for main data array
	cErr = cudaMalloc((void**)&dataMatrix_dev, get_nElements() * sizeof(float));
	checkCudaErr(cErr, "Could not allocate memory for inputData on GPU");

	// allocate memory for cumulative distribution function
	cErr = cudaMalloc((void**)&cdf_dev, nCdf * sizeof(float));
	checkCudaErr(cErr, "Could not allocate memory for cdf on GPU");

	// allocate memory for maximum values in bins
	cErr = cudaMalloc((void**)&maxValBin_dev, get_nSubVols() * sizeof(float));
	checkCudaErr(cErr, "Could not allocate memory for maxValBins on GPU");

	// allocate memory for minimum values in bins
	cErr = cudaMalloc((void**)&minValBin_dev, get_nSubVols() * sizeof(float));
	checkCudaErr(cErr, "Coult not allocate memory for miNValBins on GPU");

	// copy data matrix over
	cErr = cudaMemcpy(dataMatrix_dev, dataMatrix, get_nElements() * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErr(cErr, "Could not copy data array to GPU");

	// copy cdf
	cErr = cudaMemcpy(cdf_dev, cdf, nCdf * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErr(cErr, "Could not copy cdf to GPU");

	// copy minValBin
	cErr = cudaMemcpy(maxValBin_dev, maxValBin, get_nSubVols() * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErr(cErr, "Could not copy max val array to GPU");

	// copy maxValBin
	cErr = cudaMemcpy(minValBin_dev, minValBin, get_nSubVols() * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErr(cErr, "Could not copy min val array to GPU");

	// prepare structure with all arguments
	eq_arguments inArgsEq_h;
	#pragma unroll
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		inArgsEq_h.volSize[iDim] = volSize[iDim];
		inArgsEq_h.origin[iDim] = origin[iDim];
		inArgsEq_h.end[iDim] = endIdx[iDim];
		inArgsEq_h.nSubVols[iDim] = nSubVols[iDim];
		inArgsEq_h.spacingSubVols[iDim] = spacingSubVols[iDim];
	}

	inArgsEq_h.minValBin = minValBin_dev;
	inArgsEq_h.maxValBin = maxValBin_dev;
	inArgsEq_h.cdf = cdf_dev;
	inArgsEq_h.nBins = nBins;
	
	cErr = cudaMemcpyToSymbol(inArgsEq_d, &inArgsEq_h, sizeof(eq_arguments));
	checkCudaErr(cErr, "Could not copy symbol to GPU");

	// launch kernel
	equalize_kernel<<< gridSize, blockSize>>>(dataMatrix_dev);

	// wait for GPU calculation to finish before we copy things back
	cudaDeviceSynchronize();

	// check if there was any problem during kernel execution
	cErr = cudaGetLastError();
	checkCudaErr(cErr, "Error during eq-kernel execution");

	// copy back new data matrix
	float* ptrOutput = create_ptrOutput();
	cErr = cudaMemcpy(ptrOutput, dataMatrix_dev, get_nElements() * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErr(cErr, "Problem while copying data matrix back from device");

	// free gpu memory
	cudaFree(dataMatrix_dev);
	cudaFree(cdf_dev);
	cudaFree(maxValBin_dev);
	cudaFree(minValBin_dev);

	return;
}
#endif

// returns interpolated value between two grid positions
inline float get_interpVal(
	const float valLeft,
	const float valRight,
	const float ratio)
{
	const float interpVal = valLeft * (1.0f - ratio) + valRight * ratio;
	return interpVal;
} 

float* histeq::create_ptrOutput()
{
	float* ptrOutput;
	if (flagOverwrite)
	{
		ptrOutput = dataMatrix;
	}
	else
	{
		if (isDataOutputAlloc)
			delete[] dataOutput;

		dataOutput = new float [nElements];
		isDataOutputAlloc = 1;
		ptrOutput = dataOutput;
	}
	return ptrOutput;
}

void histeq::equalize_range(
	float* outputArray, const std::size_t idxStart, const std::size_t idxStop)
{

	std::size_t neighbours[6]; // index of next neighbouring elements
	float ratio[3]; // ratios in z x y
	float value[8];
	float ratio1[3]; // defined as 1 - ratio
	std::size_t offsetz, offsety;

	for(std::size_t iZ = idxStart; iZ <= idxStop; iZ++)
	{
		offsetz = volSize.x * volSize.y * iZ;
		for (std::size_t iY = 0; iY < volSize.y; iY++)
		{
			offsety = volSize.x * iY;
			for (std::size_t iX = 0; iX < volSize.x; iX++)
			{
				const std::size_t idxVolLin = iX + offsety + offsetz;
				const float currValue = dataMatrix[idxVolLin];
	
				const vector3<std::size_t> position = {iX, iY, iZ};
				get_neighbours(position, neighbours, ratio);

				// get values from all eight corners
				value[0] = get_icdf(neighbours[0], neighbours[2], neighbours[4], currValue);
				value[1] = get_icdf(neighbours[0], neighbours[2], neighbours[5], currValue);
				value[2] = get_icdf(neighbours[0], neighbours[3], neighbours[4], currValue);
				value[3] = get_icdf(neighbours[0], neighbours[3], neighbours[5], currValue);
				value[4] = get_icdf(neighbours[1], neighbours[2], neighbours[4], currValue);
				value[5] = get_icdf(neighbours[1], neighbours[2], neighbours[5], currValue);
				value[6] = get_icdf(neighbours[1], neighbours[3], neighbours[4], currValue);
				value[7] = get_icdf(neighbours[1], neighbours[3], neighbours[5], currValue);

				// calculate remaining ratios
				ratio1[0] = 1.0f - ratio[0];
				ratio1[1] = 1.0f - ratio[1];
				ratio1[2] = 1.0f - ratio[2];

				// trilinear interpolation
				outputArray[idxVolLin] =
					ratio1[0] * (
						ratio1[1] * (
							value[0] * ratio1[2] +
							value[1] * ratio[2]
						) + ratio[1] * (
							value[2] * ratio1[2] +
							value[3] * ratio[2] 
						)
					) + ratio[0] * (
						ratio1[1] * (
							value[4] * ratio1[2] +
							value[5] * ratio[2]
						) + ratio[1] * (
							value[6] * ratio1[2] +
							value[7] * ratio[2]
						)
					);

			}
		}
	}


	return;
}

void histeq::equalize()
{
	float* ptrOutput = create_ptrOutput();
	
	// if overwrite is disabled we need to allocate memory for new output here
	const auto tStart = std::chrono::high_resolution_clock::now();

	// calculate range for workers and launch 1 by 1
	const std::size_t zRange = volSize.z / nThreads;
	workers.clear();
	std::size_t zStart, zStop;
	for (std::size_t iWorker = 0; iWorker < nThreads; iWorker++)
	{
		zStart = zRange * iWorker;
		zStop = (iWorker == (nThreads - 1)) ? (volSize.z - 1) : (zRange * (iWorker + 1) - 1);
		std::thread currThread(&histeq::equalize_range, this, ptrOutput, zStart, zStop);
		workers.push_back(std::move(currThread));
	}

	// make sure all of our threads are ready to join back to main
	for (std::size_t iWorker = 0; iWorker < nThreads; iWorker++)
		workers[iWorker].join();

	const auto tStop = std::chrono::high_resolution_clock::now();
	const auto tDuration = std::chrono::duration_cast<std::chrono::milliseconds>(tStop- tStart);
	tEq = tDuration.count();
	return;

}



// returns a single value from our cdf function
float histeq::get_cdf(
	const std::size_t iBin, const std::size_t iXSub, const std::size_t iYSub, const std::size_t iZSub) const
{
	const std::size_t idx = iBin + nBins * (iXSub + nSubVols[0] * (iYSub +  nSubVols[1] * iZSub));
	return cdf[idx];
}

float histeq::get_cdf(const std::size_t iBin, const vector3<std::size_t> iSub) const
{
	const std::size_t idx = iBin + nBins * (iSub.x + nSubVols[0] * (iSub.y +  nSubVols[1] * iSub.z));
	return cdf[idx];
}

float histeq::get_cdf(const std::size_t iBin, const std::size_t iSubLin) const
{
	const std::size_t idx = iBin + nBins * iSubLin;
	return cdf[idx];
}

// returns a pointer to out output array (depends on flagOverwrite)
float* histeq::get_ptrOutput()
{
	if (flagOverwrite)
	{
		return dataMatrix;
	}
	else
	{
		return dataOutput;
	}
}

// returns the data value for a linearized element
float histeq::get_outputValue(const std::size_t iElem) const
{
	if (flagOverwrite)
	{
		return dataMatrix[iElem];
	}
	else
	{
		return dataOutput[iElem];
	}
} 

// returns the data value for a linearized element
float histeq::get_outputValue(const vector3<std::size_t>& idx) const
{
	const std::size_t linIdx = idx.x + volSize.x * (idx.y + volSize.y * idx.z);
	return get_outputValue(linIdx);
} 

// returns output value for 3d index
float histeq::get_outputValue(const std::size_t iZ, const std::size_t iX, const std::size_t iY) const
{
	const std::size_t linIdx = iZ + volSize[0] * (iX + volSize[1] * iY);
	return get_outputValue(linIdx);
} 

// returns the minimum value of a bin
float histeq::get_minValBin(const std::size_t zBin, const std::size_t xBin, const std::size_t yBin)
{
	const std::size_t idxBin = zBin + nSubVols[0] * (xBin + nSubVols[1] * yBin);
	return minValBin[idxBin];
}

// returns the maximum value of a bin
float histeq::get_maxValBin(const std::size_t zBin, const std::size_t xBin, const std::size_t yBin)
{
	const std::size_t idxBin = zBin + nSubVols[0] * (xBin + nSubVols[1] * yBin);
	return maxValBin[idxBin];
}

// define number of bins during eq
void histeq::set_nBins(const std::size_t _nBins)
{
	if (_nBins <= 0)
	{
		printf("The number of bins must be bigger then 0");
		throw "InvalidValue";
	}
	nBins = _nBins;
	return;
}

// define noiselevel of dataset as minimum occuring value
void histeq::set_noiseLevel(const float _noiseLevel)
{
	noiseLevel = _noiseLevel;
	return;
}

// set pointer to the data matrix
void histeq::set_data(float* _dataMatrix)
{
	dataMatrix = _dataMatrix;
	return;
}

// defines if input matrix should be overwritten or not
void histeq::set_overwrite(const bool _flagOverwrite)
{
	flagOverwrite = _flagOverwrite;
	return;
}