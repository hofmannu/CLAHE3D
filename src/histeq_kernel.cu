/*
	all kernel functions required to run CLAHE3D through CUDA
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

// const arguments passed to equilization kernel 
#ifndef EQ_ARGUMENTS_H
#define EQ_ARGUMENTS_H

struct eq_arguments
{
	uint64_t volSize[3]; // total size of volume
	float origin[3]; // origin of the subvolume grid
	float end[3]; // end of the subvolume grid
	uint64_t nSubVols[3]; // number of subvolumes
	uint64_t spacingSubVols[3]; // spacing between subvolumes
	float* minValBin; // minimum value in each bib
	float* maxValBin; // maximum value in each bin
	float* cdf; // cummulative distribution function
	uint64_t nBins; // number of bins we have for the histogram
};

#endif

// returns the indices of the neighbouring subvolumes for a defined position
__device__ inline void get_neighbours_gpu(
	const uint64_t* position,
	uint64_t* neighbours,
	float* ratio,
	const eq_arguments inArgs
	)
{
	#pragma unroll
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		// let see if we hit the lower limit
		if (((float) position[iDim]) <=  inArgs.origin[iDim])
		{
			ratio[iDim] = 0;
			neighbours[iDim * 2] = 0; // left index along current dimension
			neighbours[iDim * 2 + 1] = 0; // right index along current dimension
		}
		else if (((float) position[iDim]) >=  inArgs.end[iDim])
		{
			ratio[iDim] = 0;
			neighbours[iDim * 2] =  inArgs.nSubVols[iDim] - 1; // left index along curr dimension
		 	neighbours[iDim * 2 + 1] =   inArgs.nSubVols[iDim] - 1; // right index along curr dimension
		} 
		else // we are actually in between!
		{
			const float offsetDistance = (float) position[iDim] - (float) inArgs.origin[iDim];
			neighbours[iDim * 2] = (uint64_t) (offsetDistance / inArgs.spacingSubVols[iDim]);
			neighbours[iDim * 2 + 1] = neighbours[iDim * 2] + 1;
			const float leftDistance = offsetDistance - ((float) neighbours[iDim * 2]) * 
				((float) inArgs.spacingSubVols[iDim]);
			ratio[iDim] = leftDistance / ((float) inArgs.spacingSubVols[iDim]);
		}

	}
	return;
}

// return bin in which current value is positioned
__device__ inline float get_icdf_gpu(
	const uint64_t iZ, // index of subvolume we request along z
	const uint64_t iX, // index of subvolume we request along x
	const uint64_t iY, // index of subvolume we request along y
	const float currValue,
	const eq_arguments& inArgs)
{
	// if we are below noise level, directy return 0
	const uint64_t subVolIdx = iZ + inArgs.nSubVols[0] * (iX + inArgs.nSubVols[1] * iY);
	if (currValue <= inArgs.minValBin[subVolIdx])
	{
		return 0.0;
	}
	else
	{
		// get index describes the 3d index of the subvolume
		const uint64_t subVolOffset = inArgs.nBins * subVolIdx;
		const float vInterp = (currValue - inArgs.minValBin[subVolIdx]) / 
			(inArgs.maxValBin[subVolIdx] - inArgs.minValBin[subVolIdx]);
		
		// it can happen that the voxel value is higher then the max value detected
		// in the neighbouring histogram. In this case we crop it to the maximum permittable value
		const uint64_t binOffset = (vInterp > 1.0) ? 
			(inArgs.nBins - 1 + subVolOffset)
			: fmaf(vInterp, (float) inArgs.nBins - 1.0, 0.5) + subVolOffset;

		return inArgs.cdf[binOffset];
	}
}

// kernel function to run equilization
__global__ void equalize_kernel(
	float* dataMatrix, // input and output volume
	const eq_arguments inArgs // constant arguemtns
	)
{
	// get index of currently adjusted voxel
	const uint64_t idxVol[3] = {
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
		threadIdx.z + blockIdx.z * blockDim.z
	};

	if ( // check if we are within boundaries
		(idxVol[0] < inArgs.volSize[0]) && 
		(idxVol[1] < inArgs.volSize[1]) && 
		(idxVol[2] < inArgs.volSize[2]))
	{
		const uint64_t idxVolLin = idxVol[0] + inArgs.volSize[0] * 
			(idxVol[1] + inArgs.volSize[1] * idxVol[2]);
		const float currValue = dataMatrix[idxVolLin];

		// get neighbours defined as the subvolume indices at lower and upper end
		uint64_t neighbours[6];
		float ratio[3];
		get_neighbours_gpu(idxVol, neighbours, ratio, inArgs);
		
		// get values from all eight corners
		const float value[8] = {
			get_icdf_gpu(neighbours[0], neighbours[2], neighbours[4], currValue, inArgs),
			get_icdf_gpu(neighbours[0], neighbours[2], neighbours[5], currValue, inArgs),
			get_icdf_gpu(neighbours[0], neighbours[3], neighbours[4], currValue, inArgs),
			get_icdf_gpu(neighbours[0], neighbours[3], neighbours[5], currValue, inArgs),
			get_icdf_gpu(neighbours[1], neighbours[2], neighbours[4], currValue, inArgs),
			get_icdf_gpu(neighbours[1], neighbours[2], neighbours[5], currValue, inArgs),
			get_icdf_gpu(neighbours[1], neighbours[3], neighbours[4], currValue, inArgs),
			get_icdf_gpu(neighbours[1], neighbours[3], neighbours[5], currValue, inArgs)};
		
		// trilinear interpolation
		dataMatrix[idxVolLin] =
			fmaf(1 - ratio[0], 
				fmaf(1 - ratio[1], 
					fmaf(value[0], 1 - ratio[2], value[1] * ratio[2])
					, ratio[1] * fmaf(value[2], 1 - ratio[2], value[3] * ratio[2])
			), 
			ratio[0] * 
				fmaf(1 - ratio[1],
					fmaf(value[4], 1 - ratio[2], value[5] * ratio[2])
				, ratio[1] * fmaf(value[6], 1 - ratio[2], value[7] * ratio[2])
			));
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
	float origin[3];
};

#endif

// get start index limited by 0
__device__ inline uint64_t get_startIndex(const uint64_t zCenter, const int zRange)
{
	const uint64_t startIdx = (((int) zCenter - zRange) < 0) ? 0 : zCenter - zRange;
	return startIdx;
}

// get stop index limited by volume size
__device__ inline uint64_t get_stopIndex(const uint64_t zCenter, const int zRange, const uint64_t volSize)
{
	const uint64_t stopIdx = (((int) zCenter + zRange) >= volSize) ? volSize : zCenter + zRange;
	return stopIdx;
}

// return cummulative distribution function
__global__ void cdf_kernel(
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
			const float ctr = ((float) iSub[iDim]) * ((float) inArgs.spacingSubVols[iDim]) + inArgs.origin[iDim];
			startIdx[iDim] = get_startIndex(ctr, inArgs.range[iDim]);
			stopIdx[iDim] = get_stopIndex(ctr, inArgs.range[iDim], inArgs.volSize[iDim]);
		}
		
		// index of currently employed subvolume
		const uint64_t idxSubVol = iSub[0] + inArgs.nSubVols[0] * (iSub[1] + inArgs.nSubVols[1] * iSub[2]);
		float* localCdf = &cdf[inArgs.nBins * idxSubVol]; // histogram of subvolume, only temporarily requried
		// volume is indexed as iz + ix * nz + iy * nx * nz
		// cdf is indexed as [iBin, iZSub, iXSub, iYSub]

		// reset bins to zero before summing them up
		for (uint64_t iBin = 0; iBin < inArgs.nBins; iBin++)
			localCdf[iBin] = 0;

		// calculate local maximum and minimum
		const float firstVal = dataMatrix[
			startIdx[0] + inArgs.volSize[0] * (startIdx[1] + inArgs.volSize[1] * startIdx[2])];
		float tempMax = firstVal; // temporary variable to reduce data access
		float tempMin = firstVal;
		for (uint64_t iY = startIdx[2]; iY <= stopIdx[2]; iY++)
		{
			const uint64_t yOffset = iY * inArgs.volSize[0] * inArgs.volSize[1];
			for(uint64_t iX = startIdx[1]; iX <= stopIdx[1]; iX++)
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
		const float binRange = (tempMin == tempMax) ? 
			1 : (tempMax - tempMin) / ((float) inArgs.nBins);

		// sort values into bins which are above clipLimit
		for (uint64_t iY = startIdx[2]; iY <= stopIdx[2]; iY++)
		{
			const uint64_t yOffset = iY * inArgs.volSize[0] * inArgs.volSize[1];
			for(uint64_t iX = startIdx[1]; iX <= stopIdx[1]; iX++)
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
						if (iBin >= inArgs.nBins)
						{
							localCdf[inArgs.nBins - 1] += 1;
						}
						else
						{
							localCdf[iBin] += 1;
						}
					}
				}
			}
		}

		// calculate cummulative sum and scale along y
		float cdfTemp = 0;
		const float zeroElem = localCdf[0];
		for (uint64_t iBin = 0; iBin < inArgs.nBins; iBin++)
		{
			cdfTemp += localCdf[iBin];
			localCdf[iBin] = cdfTemp - zeroElem;
		}

		// now we scale cdf to max == 1 (first value is 0 anyway)
		const float cdfMax = localCdf[inArgs.nBins - 1];
		if (cdfMax > 0)
		{
			for (uint64_t iBin = 1; iBin < inArgs.nBins; iBin++)
			{
				localCdf[iBin] /= cdfMax;
			}
		}
		else
		{
			for (uint64_t iBin = 1; iBin < inArgs.nBins; iBin++)
			{
				localCdf[iBin] = ((float) iBin) / ((float) inArgs.nBins);
			}
		}
	}
	return;
}


