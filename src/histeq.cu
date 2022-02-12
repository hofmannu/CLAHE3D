#include "histeq.h"

// check errors of CUDA
void histeq::checkCudaErr(cudaError_t err, const char* msgErr)
{
	if (err != cudaSuccess)
	{
		printf("There was some CUDA error appearing along my way: %s\n",
			cudaGetErrorString(err));
		throw "CudaError";
	}
	return;
}

// constant stuff which we need to know during kernel execution
struct cdf_arguments
{
	uint64_t spacingSubVols[3]; // distance between subvolumes [z, x, y]
	uint64_t nSubVols[3]; // number of subvolumes [z, x, y]
	uint64_t volSize[3]; // overall size of data volume [z, x, y]
	int64_t range[3]; // range of each bin in each direction [z, x, y]
	uint64_t nBins; // number of bins which we use for our histogram
	float noiseLevel; // noise level in matrix
};

// get start index limited by 0
__device__ inline uint64_t getStartIndex(const uint64_t zCenter, const int zRange)
{
	const uint64_t startIdx = (((int) zCenter - zRange) < 0) ? 0 : zCenter - zRange;
	return startIdx;
}

// get stop index limited by volume size
__device__ inline uint64_t getStopIndex(const uint64_t zCenter, const int zRange, const uint64_t volSize)
{
	const uint64_t stopIdx = (((int) zCenter + zRange) >= volSize) ? volSize : zCenter + zRange;
	return stopIdx;
}


__global__ void get_cdf_kernel(
		float* cdf, 
		float* maxValBin, 
		float* minValBin, 
		const float* dataMatrix,
		const cdf_arguments inArgs
	)
{
	const uint64_t iSub[3] = {
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
		threadIdx.z + blockIdx.z * blockDim.z
	};

	if (
		(iSub[0] < inArgs.nSubVols[0]) && 
		(iSub[2] < inArgs.nSubVols[2]) && 
		(iSub[1] < inArgs.nSubVols[1]))
	{
		// get start and stop indices for currently used bin
		uint64_t startIdx[3];
		uint64_t stopIdx[3];
		#pragma unroll
		for (uint8_t iDim = 0; iDim < 3; iDim++)
		{
			const uint64_t ctr = iSub[iDim] * inArgs.spacingSubVols[iDim];
			startIdx[iDim] = getStartIndex(ctr, inArgs.range[iDim]);
			stopIdx[iDim] = getStopIndex(ctr, inArgs.range[iDim], inArgs.volSize[iDim]);
		}
		
		// index of currently employed subvolume
		const uint64_t idxSubVol = iSub[0] + inArgs.nSubVols[0] * (iSub[1] + inArgs.nSubVols[1] * iSub[2]);
		float* localCdf = &cdf[inArgs.nBins * idxSubVol]; // histogram of subvolume, only temporarily requried
		// volume is indexed as iz + ix * nz + iy * nx * nz
		// cdf is indexed as [iBin, iZSub, iXSub, iYSub]

		// reset bins to zero before summing them up
		for (uint64_t iBin = 0; iBin < inArgs.nBins; iBin++)
			localCdf[iBin] = 0;

		// calculate local maximum
		const float firstVal = dataMatrix[
			startIdx[0] + inArgs.volSize[0] * (startIdx[1] + inArgs.volSize[1] * startIdx[2])];
		float tempMax = firstVal; // temporary variable to reduce data access
		float tempMin = firstVal;
		for (uint64_t iY = startIdx[2]; iY <= stopIdx[2]; iY++)
		{
			const uint64_t yOffset = iY * inArgs.volSize[0] * inArgs.volSize[1];
			for(uint64_t iX = startIdx[1]; iX <= stopIdx[2]; iX++)
			{
				const uint64_t xOffset = iX * inArgs.volSize[0];
				for(uint64_t iZ = startIdx[0]; iZ <= stopIdx[0]; iZ++)
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
	
		tempMax = (tempMax < inArgs.noiseLevel) ? inArgs.noiseLevel : tempMax;
		maxValBin[idxSubVol] = tempMax;

		tempMin = (tempMin < inArgs.noiseLevel) ? inArgs.noiseLevel : tempMin;
		minValBin[idxSubVol] = tempMin;

		// calculate size of each bin
		const float binRange = (tempMin == tempMax) ? 1 : (tempMax - tempMin) / ((float) inArgs.nBins);

		uint64_t validVoxelCounter = 0;
		// sort values into bins which are above clipLimit
		for (uint64_t iY = startIdx[2]; iY <= stopIdx[2]; iY++)
		{
			const uint64_t yOffset = iY * inArgs.volSize[0] * inArgs.volSize[1];
			for(uint64_t iX = startIdx[1]; iX <= stopIdx[2]; iX++)
			{
				const uint64_t xOffset = iX * inArgs.volSize[0];
				for(uint64_t iZ = startIdx[0]; iZ <= stopIdx[0]; iZ++)
				{
					const float currVal = dataMatrix[iZ + xOffset + yOffset]; 
					// only add to histogram if above clip limit
					if (currVal >= inArgs.noiseLevel)
					{
						uint64_t iBin = (currVal - tempMin) / binRange;

						// special case for maximum values in subvolume (they gonna end up
						// one index above)
						if (iBin == inArgs.nBins)
						{
							iBin = inArgs.nBins - 1;
						}

						localCdf[iBin] += 1;
						validVoxelCounter++;
					}
				}
			}
		}

		if (validVoxelCounter == 0)
		{
			// if there was no valid voxel
			for (uint64_t iBin = 0; iBin < inArgs.nBins; iBin++)
				localCdf[iBin] = iBin / (float) inArgs.nBins;
		}
		else
		{
			// normalize so that sum of histogram is 1
			float cdfTemp = 0;
			for (uint64_t iBin = 0; iBin < inArgs.nBins; iBin++)
			{
				cdfTemp += localCdf[iBin] / (float) validVoxelCounter;
				localCdf[iBin] = cdfTemp;
			}
		}

	}
	return;

}

// same as calculate but this time running on the GPU
void histeq::calculate_gpu()
{
	calculate_nsubvols();
	getOverallMax();
	histGrid.calcSubVols();

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
	}
	inArgs.nBins = nBins;
	inArgs.noiseLevel = noiseLevel;

	float* dataMatrix_dev; // pointer to data matrix clone on GPU
	float* cdf_dev; // cumulative distribution function [iBin, izSub, ixSub, iySub]
	float* maxValBin_dev; // maximum value of current bin [izSub, ixSub, iySub]
	float* minValBin_dev; // minimum value of current bin [izSub, ixSub, iySub]
	const uint64_t nCdf = nBins * nSubVols[0] * nSubVols[1] * nSubVols[2];
	const uint64_t nSubs = nSubVols[0] * nSubVols[1] * nSubVols[2];

	cudaError_t cErr;

	// allocate memory for main data array
	cErr = cudaMalloc((void**)&dataMatrix_dev, nElements * sizeof(float) );
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
	
	// here we start the execution of the first kernel (dist function)
	get_cdf_kernel<<< gridSize, blockSize>>>(
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
	checkCudaErr(cErr, "Error during kernel execution");

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
	maxValBin = new float[nSubs];
	minValBin = new float[nSubs];
	isMaxValBinAlloc = 1;

	// copy back minimum and maximum values of the bins from our GPU
	cErr = cudaMemcpy(maxValBin, maxValBin_dev, nSubs * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErr(cErr, "Could not copy back maximum values of our bins from device");

	cErr = cudaMemcpy(minValBin, minValBin_dev, nSubs * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErr(cErr, "Could not copy back minimum values of our bins from device");

	cudaFree(dataMatrix_dev);
	cudaFree(cdf_dev);
	cudaFree(maxValBin_dev);
	cudaFree(minValBin_dev);
	return;
}


// class constructor
histeq::histeq()
{

}

histeq::~histeq()
{
	if (isCdfAlloc)
		delete[] cdf;

	if (isMaxValBinAlloc)
	{
		delete[] minValBin;
		delete[] maxValBin;
	}
}

// finds the maximum value in the whole matrix
void histeq::getOverallMax()
{
	overallMax = dataMatrix[0];
	for(uint64_t idx = 1; idx < nElements; idx++){
		if(dataMatrix[idx] > overallMax)
			overallMax = dataMatrix[idx];
	}
	return;
}

inline uint64_t histeq::getStartIdxSubVol(const uint64_t iSub, const uint8_t iDim)
{
	const int64_t centerPos = (int64_t) iSub * spacingSubVols[iDim];
	int64_t startIdx = centerPos - ((int) sizeSubVols[iDim] - 1) / 2; 
	startIdx = (startIdx < 0) ? 0 : startIdx;
	return (uint64_t) startIdx;
}

inline uint64_t histeq::getStopIdxSubVol(const uint64_t iSub, const uint8_t iDim)
{
	const int64_t centerPos = (int64_t) iSub * spacingSubVols[iDim];
	int64_t stopIdx = centerPos + ((int) sizeSubVols[iDim] - 1) / 2; 
	stopIdx = (stopIdx >= volSize[iDim]) ? (volSize[iDim] - 1) : stopIdx;
	return (uint64_t) stopIdx;
}

// cpu version of cummulative distribution function calculation
void histeq::calculate()
{
	calculate_nsubvols();
	getOverallMax();
	histGrid.calcSubVols();
	
	// allocate memory for transfer function
	if (isCdfAlloc)
		delete[] cdf;
	cdf = new float[nBins * nSubVols[0] * nSubVols[1] * nSubVols[2]];
	isCdfAlloc = 1;

	if (isMaxValBinAlloc)
	{
		delete[] maxValBin;
		delete[] minValBin;
	}
	maxValBin = new float[nSubVols[0] * nSubVols[1] * nSubVols[2]];
	minValBin = new float[nSubVols[0] * nSubVols[1] * nSubVols[2]];
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
					idxStart[iDim] = getStartIdxSubVol(idxSub[iDim], iDim);
					idxEnd[iDim] = getStopIdxSubVol(idxSub[iDim], iDim);
				}

				getCDF(idxStart[0], idxEnd[0], // zStart, zEnd
					idxStart[1], idxEnd[1], // xStart, xEnd
					idxStart[2], idxEnd[2], // yStart, yEnd
					idxSub[0], idxSub[1], idxSub[2]);
			}
		}
	}
	return;
}

// get cummulative distribution function for a certain bin 
void histeq::getCDF(
	const uint64_t zStart, const uint64_t zEnd, // z start & stop idx
	const uint64_t xStart, const uint64_t xEnd, // x start & stop idx
	const uint64_t yStart, const uint64_t yEnd, // y start & stop idx
	const uint64_t iZBin, const uint64_t iXBin, const uint64_t iYBin) // bin index
{

	const uint64_t idxSubVol = iZBin + iXBin * nSubVols[0] + iYBin * nSubVols[0] * nSubVols[1];
	float* hist = new float[nBins]; // histogram of subvolume, only temporarily requried

	// volume is indexed as iz + ix * nz + iy * nx * nz
	// cdf is indexed as [iBin, iZSub, iXSub, iYSub]

	// reset bins to zero before summing them up
	for (uint64_t iBin = 0; iBin < nBins; iBin++)
		hist[iBin] = 0;

	// calculate local maximum
	const float firstVal = dataMatrix[zStart + volSize[0] * (xStart + volSize[1] * yStart)];
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

	uint64_t validVoxelCounter = 0;
	// sort values into bins which are above clipLimit
	for (uint64_t iY = yStart; iY <= yEnd; iY++)
	{
		const uint64_t yOffset = iY * volSize[0] * volSize[1];
		for(uint64_t iX = xStart; iX <= xEnd; iX++)
		{
			const uint64_t xOffset = iX * volSize[0];
			for(uint64_t iZ = zStart; iZ <= zEnd; iZ++)
			{
				// only add to histogram if above clip limit
				if (dataMatrix[iZ + xOffset + yOffset] >= noiseLevel)
				{
					uint64_t iBin = (dataMatrix[iZ + xOffset + yOffset] - tempMin) / binRange;

					// special case for maximum values in subvolume (they gonna end up
					// one index above)
					if (iBin == nBins)
					{
						iBin = nBins - 1;
					}

					hist[iBin] += 1;
					validVoxelCounter++;
				}
			}
		}
	}

	if (validVoxelCounter == 0)
	{
		// if there was no valid voxel
		hist[0] = 1;
	}
	else
	{
		// normalize so that sum of histogram is 1
		for (uint64_t iBin = 0; iBin < nBins; iBin++)
		{
			hist[iBin] /= (float) (validVoxelCounter);
		}
	}

	
	// calculate cummulative sum and scale along y
	float cdfTemp = 0;
	const uint64_t binOffset = nBins * idxSubVol;
	for (uint64_t iBin = 0; iBin < nBins; iBin++)
	{
		cdfTemp += hist[iBin];
		cdf[binOffset + iBin] = cdfTemp;
	}

	delete[] hist; 
	return;
}

float histeq::get_icdf(
	const uint64_t iZ, // subvolume index in z 
	const uint64_t iX, // subvolume index in x
	const uint64_t iY, // subvolume index in y
	const float value) // value to extract
{
	// if we are below noise level, directy return
	const uint64_t subVolIdx = iZ + nSubVols[0] * iX + nSubVols[0] * nSubVols[1] * iY;
	if (value < minValBin[subVolIdx])
	{
		return 0;
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
			nBins - 1 + subVolOffset
			: (vInterp * ((float) nBins - 1.0) + 0.5) + subVolOffset;

		return cdf[binOffset];
	}
	
}

void histeq::equalize_gpu()
{

	return;
}

inline float getInterpVal(const float valLeft, const float valRight, const float ratio)
{
	const float interpVal = valLeft * (1 - ratio) + valRight * ratio;
	return interpVal;
} 


void histeq::equalize()
{
	uint64_t neighbours[6]; // index of next neighbouring elements
	float ratio[3]; // ratios in z x y
	float currValue; // value of position in input volume
	for(uint64_t iY = 0; iY < volSize[2]; iY++)
	{
		for (uint64_t iX = 0; iX < volSize[1]; iX++)
		{
			for (uint64_t iZ = 0; iZ < volSize[0]; iZ++)
			{
				currValue = dataMatrix[iZ + volSize[0] * (iX + volSize[1] * iY)];
	
				const uint64_t position[3] = {iZ, iX, iY};
				histGrid.getNeighbours(position, neighbours, ratio);
				
				// assign new value based on trilinear interpolation
				dataMatrix[iZ + volSize[0] * (iX + volSize[1] * iY)] =
				// first two opposing z corners
				(
					getInterpVal(
						get_icdf(neighbours[0], neighbours[2], neighbours[4], currValue),
						get_icdf(neighbours[1], neighbours[2], neighbours[4], currValue), ratio[0])

					* (1 - ratio[1]) +
				// fourth two opposing z corners
				getInterpVal(
					get_icdf(neighbours[0], neighbours[3], neighbours[5], currValue), 
					get_icdf(neighbours[1], neighbours[3], neighbours[5], currValue), ratio[0])
					* ratio[1]) * (1 - ratio[2]) +
				// second two opposing z corners
				(
					getInterpVal(
						get_icdf(neighbours[0], neighbours[3], neighbours[4], currValue),
						get_icdf(neighbours[1], neighbours[3], neighbours[4], currValue), ratio[0])
					* (1 - ratio[1]) +
				// third two opposing z corners
				
					getInterpVal(
						get_icdf(neighbours[0], neighbours[2], neighbours[5], currValue),
						get_icdf(neighbours[1], neighbours[2], neighbours[5], currValue), ratio[0])
					* ratio[1]) * ratio[2];
			}
		}
	}
	return;
}

// calculate number of subvolumes
void histeq::calculate_nsubvols(){
	// number of subvolumes
	for (unsigned char iDim = 0; iDim < 3; iDim++)
		nSubVols[iDim] = (volSize[iDim] - 2) / spacingSubVols[iDim] + 1;
	
	printf("[histeq] number of subvolumes: %ld, %ld, %ld\n", 
			nSubVols[0], nSubVols[1], nSubVols[2]);

	return;
}

// returns a single value from our cdf function
float histeq::get_cdf(const uint64_t iBin, const uint64_t iZSub, const uint64_t iXSub, const uint64_t iYSub)
{
	const uint64_t idx = iBin + nBins * (iZSub + nSubVols[0] * (iXSub +  nSubVols[1] * iYSub));
	return cdf[idx];
}

// define number of bins during eq
void histeq::setNBins(const uint64_t _nBins){
	if (_nBins == 0)
	{
		printf("The number of bins must be bigger then 0");
		throw "InvalidValue";
	}
	nBins = _nBins;
	return;
}

// define noiselevel of dataset as minimum occuring value
void histeq::setNoiseLevel(const float _noiseLevel)
{
	if (_noiseLevel < 0)
	{
		printf("The noise level should be at least 0");
		throw "InvalidValue";
	}
	noiseLevel = _noiseLevel;
	return;
}

// size of full three dimensional volume
void histeq::setVolSize(const uint64_t* _volSize){
	for(uint8_t iDim = 0; iDim < 3; iDim++)
	{
		if (_volSize[iDim] == 0)
		{
			printf("The size of the volume should be bigger then 0");
			throw "InvalidValue";
		}
		volSize[iDim] = _volSize[iDim];
	}

	nElements = volSize[0] * volSize[1] * volSize[2];
	histGrid.setVolumeSize(_volSize);

	return;
}

// set pointer to the data matrix
void histeq::setData(float* _dataMatrix){
	dataMatrix = _dataMatrix;
	return;
}

// defines the size of the individual subvolumes (lets make this uneven)
void histeq::setSizeSubVols(const uint64_t* _subVolSize){
	for(uint8_t iDim = 0; iDim < 3; iDim++)
	{
		if ((_subVolSize[iDim] % 2) == 0)
		{
			printf("Please choose the size of the subvolumes uneven");
			throw "InvalidValue";
		}
		sizeSubVols[iDim] = _subVolSize[iDim];
	}
	return;
}

// defines the spacing of the individual histogram samples
void histeq::setSpacingSubVols(const uint64_t* _spacingSubVols){
	for(uint8_t iDim = 0; iDim < 3; iDim++)
	{
		if (_spacingSubVols[iDim] == 0)
		{
			printf("The spacing of the subvolumes needs to be at least 1");
			throw "InvalidValue";
		}
		spacingSubVols[iDim] = _spacingSubVols[iDim];
	}

	// push same value over to interpolation grid
	histGrid.setGridSpacing(_spacingSubVols);
	float origin[3];
	for (unsigned char iDim = 0; iDim < 3; iDim++)
		origin[iDim] = 0.5 * (float) _spacingSubVols[iDim];
	
	histGrid.setGridOrigin(origin);

	return;
}
