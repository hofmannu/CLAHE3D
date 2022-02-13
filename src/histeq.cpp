#include "histeq.h"
#if USE_CUDA
	#include "histeq_kernel.cu" // all declarations for cuda are externalized here
#endif

histeq::histeq()
{
	// just an empty constructor so far
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
	const dim3 blockSize(32, 2, 2);
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
		inArgs.range[iDim] = (int64_t) (sizeSubVols[iDim] - 1) / 2; // range of each bin in each direction
		inArgs.origin[iDim] = origin[iDim];
	}
	inArgs.nBins = nBins;
	inArgs.noiseLevel = noiseLevel;

	float* dataMatrix_dev; // pointer to data matrix clone on GPU (read only)
	float* cdf_dev; // cumulative distribution function [iBin, izSub, ixSub, iySub]
	float* maxValBin_dev; // maximum value of current bin [izSub, ixSub, iySub]
	float* minValBin_dev; // minimum value of current bin [izSub, ixSub, iySub]
	const uint64_t nCdf = nBins * get_nSubVols();

	cudaError_t cErr;

	// allocate memory for main data array
	cErr = cudaMalloc((void**)&dataMatrix_dev, nElements * sizeof(float) );
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
	cErr = cudaMemcpy(dataMatrix_dev, dataMatrix, nElements * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErr(cErr, "Could not copy data array to GPU");
	
	// here we start the execution of the first kernel (dist function)
	cdf_kernel<<< gridSize, blockSize>>>(
		cdf_dev, // pointer to cumulative distribution function 
		maxValBin_dev, 
		minValBin_dev, 
		dataMatrix_dev,
		inArgs // struct containing all important input arguments
		);
	
	// wait for GPU calculation to finish before we copy things back
	cudaDeviceSynchronize();

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
// cpu version of cummulative distribution function calculation
void histeq::calculate_cdf()
{
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

	uint64_t idxStart[3]; // start index of current subvolume
	uint64_t idxEnd[3]; // end index of current subvolume

	// calculate histogram for each individual block
	for(uint64_t iYSub = 0; iYSub < nSubVols[2]; iYSub++) // for each z subvolume
	{
		for(uint64_t iXSub = 0; iXSub < nSubVols[1]; iXSub++) // for each y subvolume
		{
			for (uint64_t iZSub = 0; iZSub < nSubVols[0]; iZSub++) // for each x subvolume
			{
				// get stopping index
				const uint64_t idxSub[3] = {iZSub, iXSub, iYSub}; // index of current subvolume

				for(uint8_t iDim = 0; iDim < 3; iDim++)
				{
					idxStart[iDim] = get_startIdxSubVol(idxSub[iDim], iDim);
					idxEnd[iDim] = get_stopIdxSubVol(idxSub[iDim], iDim);
				}

				calculate_sub_cdf(idxStart[0], idxEnd[0], // zStart, zEnd
					idxStart[1], idxEnd[1], // xStart, xEnd
					idxStart[2], idxEnd[2], // yStart, yEnd
					idxSub[0], idxSub[1], idxSub[2]);
			}
		}
	}
	return;
}

// get cummulative distribution function for a certain bin 
void histeq::calculate_sub_cdf(
	const uint64_t zStart, const uint64_t zEnd, // z start & stop idx
	const uint64_t xStart, const uint64_t xEnd, // x start & stop idx
	const uint64_t yStart, const uint64_t yEnd, // y start & stop idx
	const uint64_t iZBin, const uint64_t iXBin, const uint64_t iYBin) // bin index
{

	const uint64_t idxSubVol = iZBin + nSubVols[0] * (iXBin + iYBin * nSubVols[1]);
	float* localCdf = &cdf[idxSubVol * nBins]; // histogram of subvolume, only temporarily requried

	// volume is indexed as iz + ix * nz + iy * nx * nz
	// cdf is indexed as [iBin, iZSub, iXSub, iYSub]

	// reset bins to zero before summing them up
	for (uint64_t iBin = 0; iBin < nBins; iBin++)
		localCdf[iBin] = 0;

	// calculate local maximum and minimum
	const float firstVal = dataMatrix[
		zStart + volSize[0] * (xStart + volSize[1] * yStart)];
	float tempMax = firstVal; // temporary variable to reduce data access
	float tempMin = firstVal;
	for (uint64_t iY = yStart; iY <= yEnd; iY++)
	{
		const uint64_t yOffset = iY * volSize[0] * volSize[1];
		for(uint64_t iX = xStart; iX <= xEnd; iX++)
		{
			const uint64_t xOffset = iX * volSize[0];
			for(uint64_t iZ = zStart; iZ <= zEnd; iZ++)
			{
				const float currVal = dataMatrix[iZ + xOffset + yOffset];
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
	
	tempMax = (tempMax < noiseLevel) ? noiseLevel : tempMax;
	maxValBin[idxSubVol] = tempMax;

	tempMin = (tempMin < noiseLevel) ? noiseLevel : tempMin;
	minValBin[idxSubVol] = tempMin;


	// calculate size of each bin
	const float binRange = (tempMin == tempMax) ? 1 : (tempMax - tempMin) / ((float) nBins);

	// sort values into bins which are above clipLimit
	for (uint64_t iY = yStart; iY <= yEnd; iY++)
	{
		const uint64_t yOffset = iY * volSize[0] * volSize[1];
		for(uint64_t iX = xStart; iX <= xEnd; iX++)
		{
			const uint64_t xOffset = iX * volSize[0];
			for(uint64_t iZ = zStart; iZ <= zEnd; iZ++)
			{
				const float currVal = dataMatrix[iZ + xOffset + yOffset];
				// only add to histogram if above clip limit
				if (currVal >= noiseLevel)
				{
					uint64_t iBin = (currVal - tempMin) / binRange;

					// special case for maximum values in subvolume (they gonna end up
					// one index above)
					if (iBin == nBins)
					{
						iBin = nBins - 1;
					}

					localCdf[iBin] += 1;
				}
			}
		}
	}

	localCdf[0] = 0; // we ignore the first bin since it is minimum anyway and should point to 0
	float cdfTemp = 0;
	for (uint64_t iBin = 1; iBin < nBins; iBin++)
	{
		cdfTemp += localCdf[iBin];
		localCdf[iBin] = cdfTemp;
	}

	// now we scale cdf to max == 1 (first value is 0 anyway)
	const float cdfMax = localCdf[nBins - 1];
	if (cdfMax > 0)
	{
		for (uint64_t iBin = 1; iBin < nBins; iBin++)
		{
			localCdf[iBin] = localCdf[iBin] / cdfMax;
		}
	}
	else
	{
		for (uint64_t iBin = 1; iBin < nBins; iBin++)
		{
			localCdf[iBin] = ((float) iBin) / ((float) nBins);
		}
	}

	return;
}

// returns the inverted cummulative distribution function
float histeq::get_icdf(
	const uint64_t iZ, // subvolume index in z 
	const uint64_t iX, // subvolume index in x
	const uint64_t iY, // subvolume index in y
	const float value) // value to extract
{
	// if we are below noise level, directy return
	const uint64_t subVolIdx = iZ + nSubVols[0] * (iX + nSubVols[1] * iY);
	if (value < minValBin[subVolIdx])
	{
		return 0.0;
	}
	else
	{
		// get index describes the 3d index of the subvolume
		const uint64_t subVolOffset = nBins * subVolIdx;
		const float vInterp = (value - minValBin[subVolIdx]) / 
			(maxValBin[subVolIdx] - minValBin[subVolIdx]); // should now span 0 to 1 		
		// it can happen that the voxel value is higher then the max value detected
		// in the next volume. In this case we crop it to the maximum permittable value
		const uint64_t binOffset = (vInterp > 1) ? 
			(nBins - 1 + subVolOffset)
			: (vInterp * ((float) nBins - 1.0) + 0.5) + subVolOffset;

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
	const uint64_t nCdf = nBins * nSubVols[0] * nSubVols[1] * nSubVols[2];
	const uint64_t nSubs = nSubVols[0] * nSubVols[1] * nSubVols[2];

	// allocate memory for main data array
	cErr = cudaMalloc((void**)&dataMatrix_dev, nElements * sizeof(float));
	checkCudaErr(cErr, "Could not allocate memory for inputData on GPU");

	// allocate memory for cumulative distribution function
	cErr = cudaMalloc((void**)&cdf_dev, nCdf * sizeof(float));
	checkCudaErr(cErr, "Could not allocate memory for cdf on GPU");

	// allocate memory for maximum values in bins
	cErr = cudaMalloc((void**)&maxValBin_dev, nSubs * sizeof(float));
	checkCudaErr(cErr, "Could not allocate memory for maxValBins on GPU");

	// allocate memory for minimum values in bins
	cErr = cudaMalloc((void**)&minValBin_dev, nSubs * sizeof(float));
	checkCudaErr(cErr, "Coult not allocate memory for miNValBins on GPU");

	// copy data matrix over
	cErr = cudaMemcpy(dataMatrix_dev, dataMatrix, nElements * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErr(cErr, "Could not copy data array to GPU");

	// copy cdf
	cErr = cudaMemcpy(cdf_dev, cdf, nCdf * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErr(cErr, "Could not copy cdf to GPU");

	// copy minValBin
	cErr = cudaMemcpy(maxValBin_dev, maxValBin, nSubs * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErr(cErr, "Could not copy max val array to GPU");

	// copy maxValBin
	cErr = cudaMemcpy(minValBin_dev, minValBin, nSubs * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErr(cErr, "Could not copy min val array to GPU");


	eq_arguments inArgs;
	#pragma unroll
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		inArgs.volSize[iDim] = volSize[iDim];
		inArgs.origin[iDim] = origin[iDim];
		inArgs.end[iDim] = end[iDim];
		inArgs.nSubVols[iDim] = nSubVols[iDim];
		inArgs.spacingSubVols[iDim] = spacingSubVols[iDim];
	}

	inArgs.minValBin = minValBin_dev;
	inArgs.maxValBin = maxValBin_dev;
	inArgs.cdf = cdf_dev;
	inArgs.nBins = nBins;
	
	// launch kernel
	equalize_kernel<<< gridSize, blockSize>>>(
		dataMatrix_dev, // pointer to cumulative distribution function 
		inArgs // struct containing all important constant input arguments
		);

	// wait for GPU calculation to finish before we copy things back
	cudaDeviceSynchronize();

	// check if there was any problem during kernel execution
	cErr = cudaGetLastError();
	checkCudaErr(cErr, "Error during eq-kernel execution");

	// copy back new data matrix
	float* ptrOutput = create_ptrOutput();
	cErr = cudaMemcpy(ptrOutput, dataMatrix_dev, nElements * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErr(cErr, "Problem while copying data matrix back from device");

	cudaFree(dataMatrix_dev);
	cudaFree(cdf_dev);
	cudaFree(maxValBin_dev);
	cudaFree(minValBin_dev);

	return;
}
#endif

// returns interpolated value between two grid positions
inline float get_interpVal(const float valLeft, const float valRight, const float ratio)
{
	const float interpVal = valLeft * (1 - ratio) + valRight * ratio;
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

void histeq::equalize()
{
	// if overwrite is disabled we need to allocate memory for new output here
	float* ptrOutput = create_ptrOutput();

	uint64_t neighbours[6]; // index of next neighbouring elements
	float ratio[3]; // ratios in z x y
	for(uint64_t iY = 0; iY < volSize[2]; iY++)
	{
		for (uint64_t iX = 0; iX < volSize[1]; iX++)
		{
			for (uint64_t iZ = 0; iZ < volSize[0]; iZ++)
			{
				const float currValue = 
					dataMatrix[iZ + volSize[0] * (iX + volSize[1] * iY)];
	
				const uint64_t position[3] = {iZ, iX, iY};
				get_neighbours(position, neighbours, ratio);

				// assign new value based on trilinear interpolation
				ptrOutput[iZ + volSize[0] * (iX + volSize[1] * iY)] =
				// first two opposing z corners
				(
					get_interpVal(
						get_icdf(neighbours[0], neighbours[2], neighbours[4], currValue),
						get_icdf(neighbours[1], neighbours[2], neighbours[4], currValue), ratio[0])

					* (1 - ratio[1]) +
				// fourth two opposing z corners
				get_interpVal(
					get_icdf(neighbours[0], neighbours[3], neighbours[5], currValue), 
					get_icdf(neighbours[1], neighbours[3], neighbours[5], currValue), ratio[0])
					* ratio[1]) * (1 - ratio[2]) +
				// second two opposing z corners
				(
					get_interpVal(
						get_icdf(neighbours[0], neighbours[3], neighbours[4], currValue),
						get_icdf(neighbours[1], neighbours[3], neighbours[4], currValue), ratio[0])
					* (1 - ratio[1]) +
				// third two opposing z corners
				
					get_interpVal(
						get_icdf(neighbours[0], neighbours[2], neighbours[5], currValue),
						get_icdf(neighbours[1], neighbours[2], neighbours[5], currValue), ratio[0])
					* ratio[1]) * ratio[2];
			}
		}
	}
	return;
}



// returns a single value from our cdf function
float histeq::get_cdf(const uint64_t iBin, const uint64_t iZSub, const uint64_t iXSub, const uint64_t iYSub)
{
	const uint64_t idx = iBin + nBins * (iZSub + nSubVols[0] * (iXSub +  nSubVols[1] * iYSub));
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
float histeq::get_outputValue(const uint64_t iElem) const
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

// returns output value for 3d index
float histeq::get_outputValue(const uint64_t iZ, const uint64_t iX, const uint64_t iY) const
{
	const uint64_t linIdx = iZ + volSize[0] * (iX + volSize[1] * iY);
	if (flagOverwrite)
	{
		return dataMatrix[linIdx];
	}
	else
	{
		return dataOutput[linIdx];
	}
} 

// returns the minimum value of a bin
float histeq::get_minValBin(const uint64_t zBin, const uint64_t xBin, const uint64_t yBin)
{
	const uint64_t idxBin = zBin + nSubVols[0] * (xBin + nSubVols[1] * yBin);
	return minValBin[idxBin];
}

// returns the maximum value of a bin
float histeq::get_maxValBin(const uint64_t zBin, const uint64_t xBin, const uint64_t yBin)
{
	const uint64_t idxBin = zBin + nSubVols[0] * (xBin + nSubVols[1] * yBin);
	return maxValBin[idxBin];
}

// define number of bins during eq
void histeq::set_nBins(const uint64_t _nBins)
{
	if (_nBins == 0)
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
	if (_noiseLevel < 0)
	{
		printf("The noise level should be at least 0");
		throw "InvalidValue";
	}
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