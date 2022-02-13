/*
	all kernel functions required go here
*/


#ifndef EQ_ARGUMENTS_H
#define EQ_ARGUMENTS_H

struct eq_arguments
{
	uint64_t volSize[3];
};

#endif

__device__ inline void get_neighbours(
	const uint64_t* position,
	uint64_t* neighbours,
	float* ratio,
	const eq_arguments inArgs
	)
{
	
	return;
}

__device__ inline float get_icdf(
	const uint64_t iZ, const uint64_t inArgs)
{
	return 0.0;	
}

__global__ void equalize_kernel(
	float* dataMatrix,
	const eq_arguments inArgs
	)
{

	const uint64_t idxVol[3] = {
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
		threadIdx.z + blockIdx.z * blockDim.z
	};

	if (
		(idxVol[0] < inArgs.volSize[0]) && 
		(idxVol[1] < inArgs.volSize[1]) && 
		(idxVol[2] < inArgs.volSize[2]))
	{
		const uint64_t idxVolLin = idxVol[0] + inArgs.volSize[0] * 
			(idxVol[1] + inArgs.volSize[1] * idxVol[2]);
		const float currValue = dataMatrix[idxVolLin];

		// now we need to get the neighbours defined as the subvolume
		// indices at lower and upper end
		uint64_t neighbours[6];
		float ratio[3];
		get_neighbours(idxVol, neighbours, ratio, inArgs);

	}

	return;
}

// struct holding arguments used in cdf kernel
#ifndef CDF_ARGUMENTS_H
#define CDF_ARGUMENTS_H

struct cdf_arguments
{
	uint64_t spacingSubVols[3]; // distance between subvolumes [z, x, y]
	uint64_t nSubVols[3]; // number of subvolumes [z, x, y]
	uint64_t volSize[3]; // overall size of data volume [z, x, y]
	int64_t range[3]; // range of each bin in each direction [z, x, y]
	uint64_t nBins; // number of bins which we use for our histogram
	float noiseLevel; // noise level in matrix
};

#endif

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


