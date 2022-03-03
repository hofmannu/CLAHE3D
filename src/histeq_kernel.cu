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
	unsigned int volSize[3]; // total size of volume
	float origin[3]; // origin of the subvolume grid
	float end[3]; // end of the subvolume grid
	unsigned int nSubVols[3]; // number of subvolumes
	unsigned int spacingSubVols[3]; // spacing between subvolumes
	float* minValBin; // minimum value in each bib
	float* maxValBin; // maximum value in each bin
	float* cdf; // cummulative distribution function
	unsigned int nBins; // number of bins we have for the histogram
};

#endif

// this struct will serve as the result of the neighbour calculation
#ifndef NEIGHBOURS_RESULT_H
#define NEIGHBOURS_RESULT_H

struct neighbour_result
{
	int neighbours[6];
	float ratio[3];
};

#endif

// struct holding arguments used in cdf kernel
#ifndef CDF_ARGUMENTS_H
#define CDF_ARGUMENTS_H

struct cdf_arguments
{
	unsigned int spacingSubVols[3]; // distance between subvolumes [z, x, y]
	unsigned int nSubVols[3]; // number of subvolumes [z, x, y]
	unsigned int volSize[3]; // overall size of data volume [z, x, y]
	unsigned int sizeSubVols[3]; // overall size of subvolume [z, x, y]
	unsigned int range[3]; // range of each bin in each direction [z, x, y]
	int nBins; // number of bins which we use for our histogram
	unsigned int noiseLevel; // noise level in matrix
	float origin[3];
};

#endif

#ifndef VECTOR3GPU_H
#define VECTOR3GPU_H

struct vector3gpu
{
	unsigned int x;
	unsigned int y;
	unsigned int z;
};

#endif

__constant__ cdf_arguments inArgsCdf[1];
__constant__ eq_arguments inArgsEq_d[1];


// returns the indices of the neighbouring subvolumes for a defined position
__device__ inline neighbour_result get_neighbours_gpu(const unsigned int* position)
{
	neighbour_result res;
	#pragma unroll
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		// let see if we hit the lower limit
		if (position[iDim] <=  inArgsEq_d->origin[iDim])
		{
			res.ratio[iDim] = 0.0f;
			res.neighbours[iDim * 2] = 0; // left index along current dimension
			res.neighbours[iDim * 2 + 1] = 0; // right index along current dimension
		}
		else if (position[iDim] >=  inArgsEq_d->end[iDim])
		{
			res.ratio[iDim] = 0.0f;
			res.neighbours[iDim * 2] =  inArgsEq_d->nSubVols[iDim] - 1; // left index along curr dimension
		 	res.neighbours[iDim * 2 + 1] =   inArgsEq_d->nSubVols[iDim] - 1; // right index along curr dimension
		} 
		else // we are actually in between!
		{
			const float offsetDistance = (float) position[iDim] - (float) inArgsEq_d->origin[iDim];
			res.neighbours[iDim * 2] = (int) (offsetDistance / inArgsEq_d->spacingSubVols[iDim]);
			res.neighbours[iDim * 2 + 1] = res.neighbours[iDim * 2] + 1;
			const float leftDistance = offsetDistance - ((float) res.neighbours[iDim * 2]) * 
				((float) inArgsEq_d->spacingSubVols[iDim]);
			res.ratio[iDim] = leftDistance / ((float) inArgsEq_d->spacingSubVols[iDim]);
		}

	}
	return res;
}

// return bin in which current value is positioned
__device__ inline float get_icdf_gpu(
	const unsigned int iZ, // index of subvolume we request along z
	const unsigned int iX, // index of subvolume we request along x
	const unsigned int iY, // index of subvolume we request along y
	const float currValue)
{
	// if we are below noise level, directy return 0
	const unsigned int subVolIdx = iZ + inArgsEq_d->nSubVols[0] * (iX + inArgsEq_d->nSubVols[1] * iY);
	if (currValue <= inArgsEq_d->minValBin[subVolIdx])
	{
		return 0.0;
	}
	else
	{
		// get index describes the 3d index of the subvolume
		const unsigned int subVolOffset = inArgsEq_d->nBins * subVolIdx;
		const float vInterp = (currValue - inArgsEq_d->minValBin[subVolIdx]) / 
			(inArgsEq_d->maxValBin[subVolIdx] - inArgsEq_d->minValBin[subVolIdx]);
		
		// it can happen that the voxel value is higher then the max value detected
		// in the neighbouring histogram. In this case we crop it to the maximum permittable value
		const unsigned int binOffset = (vInterp > 1.0f) ? 
			(inArgsEq_d->nBins - 1 + subVolOffset)
			: fmaf(vInterp, (float) inArgsEq_d->nBins - 1.0f, 0.5f) + subVolOffset;

		return inArgsEq_d->cdf[binOffset];
	}
}


// kernel function to run equilization
__global__ void equalize_kernel(float* dataMatrix)
{
	// get index of currently adjusted voxel
	const unsigned int idxVol[3] = {
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
		threadIdx.z + blockIdx.z * blockDim.z
	};

	if ( // check if we are within boundaries
		(idxVol[0] < inArgsEq_d->volSize[0]) && 
		(idxVol[1] < inArgsEq_d->volSize[1]) && 
		(idxVol[2] < inArgsEq_d->volSize[2]))
	{
		// get current value to convert
		const unsigned int idxVolLin = idxVol[0] + inArgsEq_d->volSize[0] * 
			(idxVol[1] + inArgsEq_d->volSize[1] * idxVol[2]);
		const float currValue = dataMatrix[idxVolLin];

		// get neighbours defined as the subvolume indices at lower and upper end
		const neighbour_result currRes = get_neighbours_gpu(idxVol);
		
		// get values from all eight corners
		const float value[8] = {
			get_icdf_gpu(currRes.neighbours[0], currRes.neighbours[2], currRes.neighbours[4], currValue),
			get_icdf_gpu(currRes.neighbours[0], currRes.neighbours[2], currRes.neighbours[5], currValue),
			get_icdf_gpu(currRes.neighbours[0], currRes.neighbours[3], currRes.neighbours[4], currValue),
			get_icdf_gpu(currRes.neighbours[0], currRes.neighbours[3], currRes.neighbours[5], currValue),
			get_icdf_gpu(currRes.neighbours[1], currRes.neighbours[2], currRes.neighbours[4], currValue),
			get_icdf_gpu(currRes.neighbours[1], currRes.neighbours[2], currRes.neighbours[5], currValue),
			get_icdf_gpu(currRes.neighbours[1], currRes.neighbours[3], currRes.neighbours[4], currValue),
			get_icdf_gpu(currRes.neighbours[1], currRes.neighbours[3], currRes.neighbours[5], currValue)};
		
		// trilinear interpolation
		dataMatrix[idxVolLin] =
			fmaf(1.0f - currRes.ratio[0], 
				fmaf(1.0f - currRes.ratio[1], 
					fmaf(value[0], 1.0f - currRes.ratio[2], value[1] * currRes.ratio[2])
					, currRes.ratio[1] * fmaf(value[2], 1.0f - currRes.ratio[2], value[3] * currRes.ratio[2])
			), 
			currRes.ratio[0] * 
				fmaf(1.0f - currRes.ratio[1],
					fmaf(value[4], 1.0f - currRes.ratio[2], value[5] * currRes.ratio[2])
				, currRes.ratio[1] * fmaf(value[6], 1.0f - currRes.ratio[2], value[7] * currRes.ratio[2])
			));
		}
	return;
}



// returns 3 element vector describing the starting index of our range
__device__ inline vector3gpu get_startIndex(const vector3gpu iSub)
{
	const vector3gpu centerIdx = 
	{
		iSub.x * inArgsCdf->spacingSubVols[0] + inArgsCdf->origin[0],
		iSub.y * inArgsCdf->spacingSubVols[1] + inArgsCdf->origin[1],
		iSub.z * inArgsCdf->spacingSubVols[2] + inArgsCdf->origin[2]
	};
	const vector3gpu startIdx =
	{
		((centerIdx.x - inArgsCdf->range[0]) < 0) ? 0 : centerIdx.x - inArgsCdf->range[0],
		((centerIdx.y - inArgsCdf->range[1]) < 0) ? 0 : centerIdx.y - inArgsCdf->range[1],
		((centerIdx.z - inArgsCdf->range[2]) < 0) ? 0 : centerIdx.z - inArgsCdf->range[2]
	};
	return startIdx;

}

// returns 3 element vector describing the stopping index of our range
__device__ inline vector3gpu get_stopIndex(const vector3gpu iSub)
{
	const vector3gpu centerIdx = 
	{
		iSub.x * inArgsCdf->spacingSubVols[0] + inArgsCdf->origin[0],
		iSub.y * inArgsCdf->spacingSubVols[1] + inArgsCdf->origin[1],
		iSub.z * inArgsCdf->spacingSubVols[2] + inArgsCdf->origin[2]
	};
	const vector3gpu stopIdx = 
	{
		((centerIdx.x + inArgsCdf->range[0]) < inArgsCdf->volSize[0]) ? (centerIdx.x + inArgsCdf->range[0]) : (inArgsCdf->volSize[0] - 1),
		((centerIdx.y + inArgsCdf->range[1]) < inArgsCdf->volSize[1]) ? (centerIdx.y + inArgsCdf->range[1]) : (inArgsCdf->volSize[1] - 1),
		((centerIdx.z + inArgsCdf->range[2]) < inArgsCdf->volSize[2]) ? (centerIdx.z + inArgsCdf->range[2]) : (inArgsCdf->volSize[2] - 1)
	};
	return stopIdx;
}


// return cummulative distribution function
__global__ void cdf_kernel(
		float* cdf, // cummulative distribution function [iBin, iX, iY, iZ]
		float* maxValBin, 
		float* minValBin, 
		const float* dataMatrix
	)
{
	const vector3gpu iSub = {
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
		threadIdx.z + blockIdx.z * blockDim.z
	};

	if (
		(iSub.x < inArgsCdf->nSubVols[0]) && 
		(iSub.y < inArgsCdf->nSubVols[1]) && 
		(iSub.z < inArgsCdf->nSubVols[2]))
	{
		// get start and stop indices for currently used bin
		const vector3gpu startIdx = get_startIndex(iSub);
		const vector3gpu stopIdx = get_stopIndex(iSub);
		
		// index of currently employed subvolume
		const unsigned int idxSubVol = iSub.x + inArgsCdf->nSubVols[0] * (iSub.y + inArgsCdf->nSubVols[1] * iSub.z);
		float* localCdf = &cdf[inArgsCdf->nBins * idxSubVol]; // histogram of subvolume, only temporarily requried
		// volume is indexed as iz + ix * nz + iy * nx * nz
		// cdf is indexed as [iBin, iZSub, iXSub, iYSub]

		// reset bins to zero before summing them up
		memset(&localCdf[0], 0, inArgsCdf->nBins * sizeof(float));

		// calculate local maximum and minimum
		const float firstVal = dataMatrix[
			startIdx.x + inArgsCdf->volSize[0] * (startIdx.y + inArgsCdf->volSize[1] * startIdx.z)];
		float tempMax = firstVal; // temporary variable to reduce data access
		float tempMin = firstVal;
		for (unsigned int iZ = startIdx.z; iZ <= stopIdx.z; iZ++)
		{
			const unsigned int zOffset = iZ * inArgsCdf->volSize[0] * inArgsCdf->volSize[1];
			for(unsigned int iY = startIdx.y; iY <= stopIdx.y; iY++)
			{
				const unsigned int yOffset = iY * inArgsCdf->volSize[0];
				for(unsigned int iX = startIdx.x; iX <= stopIdx.x; iX++)
				{
					const float currVal = dataMatrix[iX + zOffset + yOffset];
					
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

		tempMax = (tempMax < inArgsCdf->noiseLevel) ? inArgsCdf->noiseLevel : tempMax;
		maxValBin[idxSubVol] = tempMax;

		tempMin = (tempMin < inArgsCdf->noiseLevel) ? inArgsCdf->noiseLevel : tempMin;
		minValBin[idxSubVol] = tempMin;

		// calculate size of each bin
		const float binRange = (tempMin == tempMax) ? 
			1.0f : (tempMax - tempMin) / ((float) inArgsCdf->nBins);

		// sort values into bins which are above clipLimit
		for (unsigned int iZ = startIdx.z; iZ <= stopIdx.z; iZ++)
		{
			const int zOffset = iZ * inArgsCdf->volSize[0] * inArgsCdf->volSize[1];
			for(unsigned int iY = startIdx.y; iY <= stopIdx.y; iY++)
			{
				const unsigned int yOffset = iY * inArgsCdf->volSize[0];
				for(unsigned int iX = startIdx.x; iX <= stopIdx.x; iX++)
				{
					const float currVal = dataMatrix[iX + yOffset + zOffset]; 
					// only add to histogram if above clip limit
					if (currVal >= inArgsCdf->noiseLevel)
					{
						const unsigned int iBin = (currVal - tempMin) / binRange;

						// special case for maximum values in subvolume (they gonna end up
						// one index above)
						if (iBin >= inArgsCdf->nBins)
						{
							localCdf[inArgsCdf->nBins - 1] += 1.0f;
						}
						else
						{
							localCdf[iBin] += 1.0f;
						}
					}
				}
			}
		}

		// calculate cummulative sum and scale along y
		float cdfTemp = 1.0f;
		const float zeroElem = localCdf[0];
		for (unsigned int iBin = 0; iBin < inArgsCdf->nBins; iBin++)
		{
			cdfTemp += localCdf[iBin];
			localCdf[iBin] = cdfTemp - zeroElem;
		}

		// now we scale cdf to max == 1 (first value is 0 anyway)
		const float cdfMax = localCdf[inArgsCdf->nBins - 1];
		const float rcdfMax = 1.0f / cdfMax;
		const float revMaxBin = 1.0f / ((float) (inArgsCdf->nBins - 1));
		if (cdfMax > 0.0f)
		{
			for (unsigned int iBin = 1; iBin < inArgsCdf->nBins; iBin++)
			{
				localCdf[iBin] *= rcdfMax;
			}
		}
		else
		{
			for (unsigned int iBin = 1; iBin < inArgsCdf->nBins; iBin++)
			{
				localCdf[iBin] = ((float) iBin) * revMaxBin;
			}
		}
	}
	return;
}


