/*
	Contrast Limited Adaptive Histogram Eq

	Warning: please find the project with unit tests, actual file structure, get/set tests etc here:
		https://github.com/hofmannu/CLAHE3D

	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#include <iostream>
#include <cstdint>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// structs and datatzpes
struct eq_arguments
{
	uint64_t volSize[3]; // total size of volume
	uint64_t origin[3]; // origin of the subvolume grid
	uint64_t endIdx[3]; // end of the subvolume grid
	uint64_t nSubVols[3]; // number of subvolumes
	uint64_t spacingSubVols[3]; // spacing between subvolumes
	uint64_t nBins; // number of bins we have for the histogram
	float* minValBin; // minimum value in each bin
	float* maxValBin; // maximum value in each bin
	float* cdf; // cummulative distribution function
};

struct vector3gpu
{
	uint64_t x;
	uint64_t y;
	uint64_t z;
};

struct neighbour_result
{
	uint64_t neighbours[6];
	float ratio[3];
};

struct cdf_arguments // struct holding constant arguments used in cdf kernel
{
	uint64_t spacingSubVols[3]; // distance between subvolumes [z, x, y]
	uint64_t nSubVols[3]; // number of subvolumes [z, x, y]
	uint64_t volSize[3]; // overall size of data volume [z, x, y]
	uint64_t range[3]; // range of each bin in each direction [z, x, y]
	uint64_t nBins; // number of bins which we use for our histogram
	float noiseLevel; // noise level in matrix
	uint64_t origin[3];
};

// static memory
__constant__ cdf_arguments inArgsCdf_d[1];
__constant__ eq_arguments inArgsEq_d[1];

// returns the indices of the neighbouring subvolumes for a defined position
__device__ inline neighbour_result get_neighbours_gpu(const uint64_t* position)
{
	neighbour_result res;
	#pragma unroll
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		// let see if we hit the lower limit
		if (((float) position[iDim]) <=  inArgsEq_d->origin[iDim])
		{
			res.ratio[iDim] = 0;
			res.neighbours[iDim * 2] = 0; // left index along current dimension
			res.neighbours[iDim * 2 + 1] = 0; // right index along current dimension
		}
		else if (((float) position[iDim]) >= inArgsEq_d->endIdx[iDim])
		{
			res.ratio[iDim] = 0;
			res.neighbours[iDim * 2] =  inArgsEq_d->nSubVols[iDim] - 1; // left index along curr dimension
		 	res.neighbours[iDim * 2 + 1] =   inArgsEq_d->nSubVols[iDim] - 1; // right index along curr dimension
		} 
		else // we are actually in between!
		{
			const float offsetDistance = (float) position[iDim] - (float) inArgsEq_d->origin[iDim];
			res.neighbours[iDim * 2] = (uint64_t) (offsetDistance / inArgsEq_d->spacingSubVols[iDim]);
			res.neighbours[iDim * 2 + 1] = res.neighbours[iDim * 2] + 1;
			const float leftDistance = offsetDistance - ((float) res.neighbours[iDim * 2]) * 
				((float) inArgsEq_d->spacingSubVols[iDim]);
			res.ratio[iDim] = leftDistance / ((float) inArgsEq_d->spacingSubVols[iDim]);
		}

	}
	return res;
}

// return inverted cumulative distribution function values
__device__ inline float get_icdf_gpu(
	const uint64_t iX, // index of subvolume we request along z
	const uint64_t iY, // index of subvolume we request along x
	const uint64_t iZ, // index of subvolume we request along y
	const float currValue)
{
	// if we are below noise level, directy return 0
	const uint64_t subVolIdx = iX + inArgsEq_d->nSubVols[0] * (iY + inArgsEq_d->nSubVols[1] * iZ);
	if (currValue <= inArgsEq_d->minValBin[subVolIdx])
	{
		return 0.0;
	}
	else
	{
		// get index describes the 3d index of the subvolume
		const uint64_t subVolOffset = inArgsEq_d->nBins * subVolIdx;
		const float vInterp = (currValue - inArgsEq_d->minValBin[subVolIdx]) / 
			(inArgsEq_d->maxValBin[subVolIdx] - inArgsEq_d->minValBin[subVolIdx]); // should now span 0 to 1 
		
		// it can happen that the voxel value is higher then the max value detected
		// in the next volume. In this case we crop it to the maximum permittable value
		const uint64_t binOffset = (vInterp > 1) ? 
			(inArgsEq_d->nBins - 1 + subVolOffset)
			: (vInterp * ((float) inArgsEq_d->nBins - 1.0) + 0.5) + subVolOffset;

		return inArgsEq_d->cdf[binOffset];
	}
}

// kernel function to run equilization
__global__ void equalize_kernel(float* dataMatrix)
{
	// get index of currently adjusted voxel
	const uint64_t idxVol[3] = {
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
		const uint64_t idxVolLin = idxVol[0] + inArgsEq_d->volSize[0] * 
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
		
		// trilinear uint64_terpolation
		dataMatrix[idxVolLin] =
			fmaf(1 - currRes.ratio[0], 
				fmaf(1 - currRes.ratio[1], 
					fmaf(value[0], 1 - currRes.ratio[2], value[1] * currRes.ratio[2])
					, currRes.ratio[1] * fmaf(value[2], 1 - currRes.ratio[2], value[3] * currRes.ratio[2])
			), 
			currRes.ratio[0] * 
				fmaf(1 - currRes.ratio[1],
					fmaf(value[4], 1 - currRes.ratio[2], value[5] * currRes.ratio[2])
				, currRes.ratio[1] * fmaf(value[6], 1 - currRes.ratio[2], value[7] * currRes.ratio[2])
			));
		}
	return;
}

// returns 3 element vector describing the starting index of our range
__device__ inline vector3gpu get_startIndex(const vector3gpu iSub)
{
	const vector3gpu centerIdx = 
	{
		iSub.x * inArgsCdf_d->spacingSubVols[0] + inArgsCdf_d->origin[0],
		iSub.y * inArgsCdf_d->spacingSubVols[1] + inArgsCdf_d->origin[1],
		iSub.z * inArgsCdf_d->spacingSubVols[2] + inArgsCdf_d->origin[2]
	};
	const vector3gpu startIdx =
	{
		(centerIdx.x < inArgsCdf_d->range[0]) ? 0 : centerIdx.x - inArgsCdf_d->range[0],
		(centerIdx.y < inArgsCdf_d->range[1]) ? 0 : centerIdx.y - inArgsCdf_d->range[1],
		(centerIdx.z < inArgsCdf_d->range[2]) ? 0 : centerIdx.z - inArgsCdf_d->range[2]
	};
	return startIdx;
}

// returns 3 element vector describing the stopping index of our range
__device__ inline vector3gpu get_stopIndex(const vector3gpu iSub)
{
	const vector3gpu centerIdx = 
	{
		iSub.x * inArgsCdf_d->spacingSubVols[0] + inArgsCdf_d->origin[0],
		iSub.y * inArgsCdf_d->spacingSubVols[1] + inArgsCdf_d->origin[1],
		iSub.z * inArgsCdf_d->spacingSubVols[2] + inArgsCdf_d->origin[2]
	};
	const vector3gpu stopIdx = 
	{
		((centerIdx.x + inArgsCdf_d->range[0]) < inArgsCdf_d->volSize[0]) ? 
			(centerIdx.x + inArgsCdf_d->range[0]) : (inArgsCdf_d->volSize[0] - 1),
		((centerIdx.y + inArgsCdf_d->range[1]) < inArgsCdf_d->volSize[1]) ? 
			(centerIdx.y + inArgsCdf_d->range[1]) : (inArgsCdf_d->volSize[1] - 1),
		((centerIdx.z + inArgsCdf_d->range[2]) < inArgsCdf_d->volSize[2]) ? 
			(centerIdx.z + inArgsCdf_d->range[2]) : (inArgsCdf_d->volSize[2] - 1)
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
		(iSub.x < inArgsCdf_d->nSubVols[0]) && 
		(iSub.y < inArgsCdf_d->nSubVols[1]) && 
		(iSub.z < inArgsCdf_d->nSubVols[2]))
	{
		// get start and stop indices for currently used bin
		const vector3gpu startIdx = get_startIndex(iSub);
		const vector3gpu stopIdx = get_stopIndex(iSub);
		
		// index of currently employed subvolume
		const uint64_t idxSubVol = iSub.x + inArgsCdf_d->nSubVols[0] * (iSub.y + inArgsCdf_d->nSubVols[1] * iSub.z);
		float* localCdf = &cdf[inArgsCdf_d->nBins * idxSubVol]; // histogram of subvolume, only temporarily requried
		// volume is indexed as iz + ix * nz + iy * nx * nz
		// cdf is indexed as [iBin, iZSub, iXSub, iYSub]

		memset(&localCdf[0], 0, inArgsCdf_d->nBins * sizeof(float));

		// calculate local maximum and minimum
		const float firstVal = dataMatrix[
			startIdx.x + inArgsCdf_d->volSize[0] * (startIdx.y + inArgsCdf_d->volSize[1] * startIdx.z)];
		float tempMax = firstVal; // temporary variable to reduce data access
		float tempMin = firstVal;
		for (uint64_t iZ = startIdx.z; iZ <= stopIdx.z; iZ++)
		{
			const uint64_t zOffset = iZ * inArgsCdf_d->volSize[0] * inArgsCdf_d->volSize[1];
			for(uint64_t iY = startIdx.y; iY <= stopIdx.y; iY++)
			{
				const uint64_t yOffset = iY * inArgsCdf_d->volSize[0];
				for(uint64_t iX = startIdx.x; iX <= stopIdx.x; iX++)
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

		tempMax = (tempMax < inArgsCdf_d->noiseLevel) ? inArgsCdf_d->noiseLevel : tempMax;
		maxValBin[idxSubVol] = tempMax;

		tempMin = (tempMin < inArgsCdf_d->noiseLevel) ? inArgsCdf_d->noiseLevel : tempMin;
		minValBin[idxSubVol] = tempMin;

		// calculate size of each bin
		const float binRange = (tempMin == tempMax) ? 
			1 : (tempMax - tempMin) / ((float) inArgsCdf_d->nBins);

		// sort values uint64_to bins which are above clipLimit
		for (uint64_t iZ = startIdx.z; iZ <= stopIdx.z; iZ++)
		{
			const uint64_t zOffset = iZ * inArgsCdf_d->volSize[0] * inArgsCdf_d->volSize[1];
			for(uint64_t iY = startIdx.y; iY <= stopIdx.y; iY++)
			{
				const uint64_t yOffset = iY * inArgsCdf_d->volSize[0];
				for(uint64_t iX = startIdx.x; iX <= stopIdx.x; iX++)
				{
					const float currVal = dataMatrix[iX + yOffset + zOffset]; 
					// only add to histogram if above clip limit
					if (currVal >= inArgsCdf_d->noiseLevel)
					{
						const uint64_t iBin = (currVal - tempMin) / binRange;

						// special case for maximum values in subvolume (they gonna end up
						// one index above)
						if (iBin >= inArgsCdf_d->nBins)
						{
							localCdf[inArgsCdf_d->nBins - 1] += 1;
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
		for (uint64_t iBin = 0; iBin < inArgsCdf_d->nBins; iBin++)
		{
			cdfTemp += localCdf[iBin];
			localCdf[iBin] = cdfTemp - zeroElem;
		}

		// now we scale cdf to max == 1 (first value is 0 anyway)
		const float cdfMax = localCdf[inArgsCdf_d->nBins - 1];
		if (cdfMax > 0)
		{
			for (uint64_t iBin = 1; iBin < inArgsCdf_d->nBins; iBin++)
			{
				localCdf[iBin] /= cdfMax;
			}
		}
		else
		{
			for (uint64_t iBin = 1; iBin < inArgsCdf_d->nBins; iBin++)
			{
				localCdf[iBin] = ((float) iBin) / ((float) inArgsCdf_d->nBins - 1);
			}
		}
	}
	return;
}

// definition of  the grid used for our histogram calculations
	
class gridder
{
public:
	// variables
	uint64_t volSize[3] = {600, 500, 400}; // size of full three dimensional volume [z, x, y]
	uint64_t nSubVols[3]; // number of subvolumes in zxy
	uint64_t sizeSubVols[3] = {31, 31, 31}; // size of each subvolume in zxy (should be uneven)
	uint64_t spacingSubVols[3] = {10, 10, 10}; // spacing of subvolumes (they can overlap)
	uint64_t origin[3]; // position of the very first element [0, 0, 0]
	uint64_t endIdx[3]; // terminal value
	uint64_t nElements; // total number of elements
	
	// get functions
	void calculate_nsubvols();
	uint64_t get_nSubVols(const uint8_t iDim) const {return nSubVols[iDim];};
	uint64_t get_nSubVols() const {return nSubVols[0] * nSubVols[1] * nSubVols[2];};
	uint64_t get_nElements() const {return volSize[0] * volSize[1] * volSize[2];};
};


// calculate number of subvolumes
void gridder::calculate_nsubvols()
{
	// number of subvolumes
	#pragma unroll
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		const uint64_t lastIdx = volSize[iDim] - 1;
		origin[iDim] = (sizeSubVols[iDim] - 1) / 2;
		nSubVols[iDim] = (lastIdx - origin[iDim]) / spacingSubVols[iDim] + 1;
		endIdx[iDim] = origin[iDim] + (nSubVols[iDim] - 1) * spacingSubVols[iDim];
	}
	return;
}

// small helper class to show errors
class cudaTools
{
	public:
		cudaError_t cErr;
	
		void checkCudaErr(cudaError_t err, const char* msgErr)
		{
			if (err != cudaSuccess)
			{
				printf("There was some CUDA error: %s, %s\n",
					msgErr, cudaGetErrorString(err));
				throw "CudaError";
			}
			return;
		};
};


// main class used for histogram equ
class histeq: public cudaTools, public gridder
{
	public:
		float* dataMatrix; // 3d matrix containing input and output volume
		uint64_t nBins = 20; // number of histogram bins
		float noiseLevel = 0.1; // noise level threshold (clipLimit)
		void calculate_gpu();
};

// same as calculate but this time running on the GPU
void histeq::calculate_gpu()
{
	calculate_nsubvols();

	printf("Starting CLAHE3D execution...\n");
		
	const dim3 blockSize(32, 2, 2); // define grid and block size
	const dim3 gridSize(
		(nSubVols[0] + blockSize.x - 1) / blockSize.x,
		(nSubVols[1] + blockSize.y - 1) / blockSize.y,
		(nSubVols[2] + blockSize.z - 1) / blockSize.z);

	// prepare input argument struct
	printf("Starting CDF...\n");
	
	cdf_arguments inArgsCdf_h;
	#pragma unroll
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		inArgsCdf_h.spacingSubVols[iDim] = spacingSubVols[iDim]; // size of each subvolume
		inArgsCdf_h.nSubVols[iDim] = nSubVols[iDim]; // number of subvolumes
		inArgsCdf_h.volSize[iDim] = volSize[iDim]; // overall size of data volume
		inArgsCdf_h.range[iDim] = (sizeSubVols[iDim] - 1) / 2; // range of each bin in each direction
		inArgsCdf_h.origin[iDim] = origin[iDim];
	}
	inArgsCdf_h.nBins = nBins;
	inArgsCdf_h.noiseLevel = noiseLevel;
	
	float* dataMatrix_dev; // pouint64_ter to data matrix clone on GPU (read only)
	float* cdf_dev; // cumulative distribution function [iBin, izSub, ixSub, iySub]
	float* maxValBin_dev; // maximum value of current bin [izSub, ixSub, iySub]
	float* minValBin_dev; // minimum value of current bin [izSub, ixSub, iySub]
	const uint64_t nCdf = nBins * get_nSubVols();

	cudaError_t cErr;

	// allocate memory for main data array
	cErr = cudaMalloc((void**)&dataMatrix_dev, get_nElements() * sizeof(float) );
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
	
	cErr = cudaMemcpyToSymbol(inArgsCdf_d, &inArgsCdf_h, sizeof(cdf_arguments));
	checkCudaErr(cErr, "Could not copy symbol to GPU");

	// here we start the execution of the first kernel (dist function)
	cdf_kernel<<< gridSize, blockSize>>>(
		cdf_dev, // pouint64_ter to cumulative distribution function 
		maxValBin_dev, 
		minValBin_dev, 
		dataMatrix_dev
	);
	
	// wait for GPU calculation to finish before we copy things back
	cudaDeviceSynchronize();
	
	// check if there was any problem during kernel execution
	cErr = cudaGetLastError();
	checkCudaErr(cErr, "Error during cdf-kernel execution");

	eq_arguments inArgsEq_h;
	#pragma unroll
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		inArgsEq_h.volSize[iDim] = volSize[iDim];
		inArgsEq_h.origin[iDim] = origin[iDim];
		inArgsEq_h.endIdx[iDim] = endIdx[iDim];
		inArgsEq_h.nSubVols[iDim] = nSubVols[iDim];
		inArgsEq_h.spacingSubVols[iDim] = spacingSubVols[iDim];
		// inArgsEq_h.sizeSubVols[iDim] = sizeSubVols[iDim];
	}
	inArgsEq_h.minValBin = minValBin_dev;
	inArgsEq_h.maxValBin = maxValBin_dev;
	inArgsEq_h.cdf = cdf_dev;
	inArgsEq_h.nBins = nBins;

	printf("Starting EQ...\n");
	
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
	cErr = cudaMemcpy(dataMatrix, dataMatrix_dev, get_nElements() * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErr(cErr, "Problem while copying data matrix back from device");
	
	cudaFree(dataMatrix_dev);
	cudaFree(cdf_dev);
	cudaFree(maxValBin_dev);
	cudaFree(minValBin_dev);
	return;
}

int main()
{
	srand(1);
	// generate input volume matrix and assign random values to it
	histeq histHandler;

	// fill matrix with random values
	histHandler.dataMatrix = new float[histHandler.get_nElements()];
	for(uint64_t iIdx = 0; iIdx < histHandler.get_nElements(); iIdx++)
		histHandler.dataMatrix[iIdx] = ((float) rand()) / ((float) RAND_MAX);

	// run calculation
	histHandler.calculate_gpu();

	delete[] histHandler.dataMatrix;
	return 0;
}
