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
#include <fstream>
#include <chrono>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// arguments structure passed to equilization kernel 
struct eq_arguments
{
	int volSize[3]; // total size of volume
	float origin[3]; // origin of the subvolume grid
	float end[3]; // end of the subvolume grid
	int nSubVols[3]; // number of subvolumes
	int spacingSubVols[3]; // spacing between subvolumes
	float* minValBin; // minimum value in each bib
	float* maxValBin; // maximum value in each bin
	float* cdf; // cummulative distribution function
	int nBins; // number of bins we have for the histogram
};

// returns the indices of the neighbouring subvolumes for a defined position
__device__ inline void get_neighbours_gpu(
	const int* position,
	int* neighbours,
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
			const float offsetDistance = ((float) position[iDim]) - (float) inArgs.origin[iDim];
			neighbours[iDim * 2] = (int) (offsetDistance / inArgs.spacingSubVols[iDim]);
			neighbours[iDim * 2 + 1] = neighbours[iDim * 2] + 1;
			const float leftDistance = offsetDistance - ((float) neighbours[iDim * 2]) * 
				inArgs.spacingSubVols[iDim];
			ratio[iDim] = leftDistance / ((float) inArgs.spacingSubVols[iDim]);
		}

	}
	return;
}

// return inverted cumulative distribution function values
__device__ inline float get_icdf_gpu(
	const int iZ, // index of subvolume we request along z
	const int iX, // index of subvolume we request along x
	const int iY, // index of subvolume we request along y
	const float currValue,
	const eq_arguments inArgs)
{
	// if we are below noise level, directy return 0
	const int subVolIdx = iZ + inArgs.nSubVols[0] * (iX + inArgs.nSubVols[1] * iY);
	if (currValue <= inArgs.minValBin[subVolIdx])
	{
		return 0;
	}
	else
	{
		// get index describes the 3d index of the subvolume
		const int subVolOffset = inArgs.nBins * subVolIdx;
		const float vInterp = (currValue - inArgs.minValBin[subVolIdx]) / 
			(inArgs.maxValBin[subVolIdx] - inArgs.minValBin[subVolIdx]); // should now span 0 to 1 
		
		// it can happen that the voxel value is higher then the max value detected
		// in the next volume. In this case we crop it to the maximum permittable value
		const int binOffset = (vInterp > 1) ? 
			(inArgs.nBins - 1 + subVolOffset)
			: (vInterp * ((float) inArgs.nBins - 1.0) + 0.5) + subVolOffset;

		return inArgs.cdf[binOffset];
	}
}

// simple interpolation between two neightbouring voxels
__device__ inline float getInterpVal_gpu(
	const float valLeft, // value of voxel on the left 
	const float valRight, // value of vocel on the right
	const float ratio) // ratio between them
{
	const float interpVal = valLeft * (1 - ratio) + valRight * ratio;
	return interpVal;
} 

// kernel function to run equilization
__global__ void equalize_kernel(
	float* dataMatrix, // input and output volume
	const eq_arguments inArgs // constant arguemtns
	)
{
	// get index of currently adjusted voxel
	const int idxVol[3] = {
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
		threadIdx.z + blockIdx.z * blockDim.z
	};

	if ( // check if we are within boundaries
		(idxVol[0] < inArgs.volSize[0]) && 
		(idxVol[1] < inArgs.volSize[1]) && 
		(idxVol[2] < inArgs.volSize[2]))
	{
		const int idxVolLin = idxVol[0] + inArgs.volSize[0] * 
			(idxVol[1] + inArgs.volSize[1] * idxVol[2]);
		const float currValue = dataMatrix[idxVolLin];

		// get neighbours defined as the subvolume indices at lower and upper end
		int neighbours[6];
		float ratio[3];
		get_neighbours_gpu(idxVol, neighbours, ratio, inArgs);

		dataMatrix[idxVolLin] = // assign new value based on trilinear interpolation
			(
				getInterpVal_gpu(
					get_icdf_gpu(neighbours[0], neighbours[2], neighbours[4], currValue, inArgs),
					get_icdf_gpu(neighbours[1], neighbours[2], neighbours[4], currValue, inArgs), ratio[0])
				* (1 - ratio[1]) +
				getInterpVal_gpu(
					get_icdf_gpu(neighbours[0], neighbours[3], neighbours[5], currValue, inArgs), 
					get_icdf_gpu(neighbours[1], neighbours[3], neighbours[5], currValue, inArgs), ratio[0])
				* ratio[1]) * (1 - ratio[2]) +
			(
				getInterpVal_gpu(
					get_icdf_gpu(neighbours[0], neighbours[3], neighbours[4], currValue, inArgs),
					get_icdf_gpu(neighbours[1], neighbours[3], neighbours[4], currValue, inArgs), ratio[0])
				* (1 - ratio[1]) +
				getInterpVal_gpu(
					get_icdf_gpu(neighbours[0], neighbours[2], neighbours[5], currValue, inArgs),
					get_icdf_gpu(neighbours[1], neighbours[2], neighbours[5], currValue, inArgs), ratio[0])
				* ratio[1]) * ratio[2];
		}
	return;
}

// struct holding arguments used in cdf kernel
struct cdf_arguments
{
	int spacingSubVols[3]; // distance between subvolumes [z, x, y]
	int nSubVols[3]; // number of subvolumes [z, x, y]
	int volSize[3]; // overall size of data volume [z, x, y]
	int range[3]; // range of each bin in each direction [z, x, y]
	int nBins; // number of bins which we use for our histogram
	float noiseLevel; // noise level in matrix
	float origin[3];
};

// get start index limited by 0
__device__ inline int get_startIndex(const int zCenter, const int zRange)
{
	const int startIdx = (((int) zCenter - zRange) < 0) ? 0 : zCenter - zRange;
	return startIdx;
}

// get stop index limited by volume size
__device__ inline int get_stopIndex(const int zCenter, const int zRange, const int volSize)
{
	const int stopIdx = (((int) zCenter + zRange) >= volSize) ? volSize : zCenter + zRange;
	return stopIdx;
}

// calculate cummulative distribution function for subvolumes
__global__ void cdf_kernel(
		float* cdf, // cummulative distribution functon [ibin, izb, ixb, iyb]
		float* maxValBin, // maximum value in each bin [izb, ixb, iyb]
		float* minValBin, // minimum value in each bin [izb, ixb, iyb]
		const float* dataMatrix, // data matrix
		const cdf_arguments inArgs // constant arguments such as matrix size
	)
{
	const int iSub[3] = { // get index of current subvolume
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
		threadIdx.z + blockIdx.z * blockDim.z
	};

	if ( // check if current subvolume is within bounds
		(iSub[0] < inArgs.nSubVols[0]) && 
		(iSub[1] < inArgs.nSubVols[1]) && 
		(iSub[2] < inArgs.nSubVols[2]))
	{
		// get start and stop indices for currently used bin
		int startIdx[3];	int stopIdx[3];
		#pragma unroll
		for (uint8_t iDim = 0; iDim < 3; iDim++)
		{
			const int ctr = iSub[iDim] * inArgs.spacingSubVols[iDim] + inArgs.origin[iDim];
			startIdx[iDim] = get_startIndex(ctr, inArgs.range[iDim]);
			stopIdx[iDim] = get_stopIndex(ctr, inArgs.range[iDim], inArgs.volSize[iDim]);
		}
		
		// index of currently calculated subvolume
		const int idxSubVol = iSub[0] + inArgs.nSubVols[0] * (iSub[1] + inArgs.nSubVols[1] * iSub[2]);
		float* localCdf = &cdf[inArgs.nBins * idxSubVol]; // returns our part of array

		for (int iBin = 0; iBin < inArgs.nBins; iBin++) // reset bins
			localCdf[iBin] = 0;

		// calculate local maximum and minimum
		const float firstVal = dataMatrix[
			startIdx[0] + inArgs.volSize[0] * (startIdx[1] + inArgs.volSize[1] * startIdx[2])];
		float tempMax = firstVal; // temporary variable to reduce data access
		float tempMin = firstVal;
		for (int iY = startIdx[2]; iY <= stopIdx[2]; iY++)
		{
			const int yOffset = iY * inArgs.volSize[0] * inArgs.volSize[1];
			for(int iX = startIdx[1]; iX <= stopIdx[1]; iX++)
			{
				const int xOffset = iX * inArgs.volSize[0];
				for(int iZ = startIdx[0]; iZ <= stopIdx[0]; iZ++)
				{
					const float currVal = dataMatrix[iZ + xOffset + yOffset];
					
					if (currVal > tempMax)
						tempMax = currVal;
					
					if (currVal < tempMin)
						tempMin = currVal;
					
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
		for (int iY = startIdx[2]; iY <= stopIdx[2]; iY++)
		{
			const int yOffset = iY * inArgs.volSize[0] * inArgs.volSize[1];
			for(int iX = startIdx[1]; iX <= stopIdx[1]; iX++)
			{
				const int xOffset = iX * inArgs.volSize[0];
				for(int iZ = startIdx[0]; iZ <= stopIdx[0]; iZ++)
				{
					const float currVal = dataMatrix[iZ + xOffset + yOffset]; 
					// only add to histogram if above clip limit
					if (currVal >= inArgs.noiseLevel)
					{
						int iBin = (currVal - tempMin) / binRange;

						// special case for maximum values in subvolume (they gonna end up
						// one index above)
						if (iBin >= inArgs.nBins)
						{
							iBin = inArgs.nBins - 1;
						}

						localCdf[iBin] += 1;
					}
				}
			}
		}

		// calculate cummulative sum and scale along y
		localCdf[0] = 0; // we ignore the first bin since it is minimum anyway and should point to 0
		float cdfTemp = 0;
		for (int iBin = 1; iBin < inArgs.nBins; iBin++)
		{
			cdfTemp += localCdf[iBin];
			localCdf[iBin] = cdfTemp;
		}

		// now we scale cdf to max == 1 (first value is 0 anyway)
		const float cdfMax = localCdf[inArgs.nBins - 1];
		if (cdfMax > 0)
		{
			for (int iBin = 1; iBin < inArgs.nBins; iBin++)
			{
				localCdf[iBin] = localCdf[iBin] / cdfMax;
			}
		}
		else
		{
			for (int iBin = 1; iBin < inArgs.nBins; iBin++)
			{
				localCdf[iBin] = ((float) iBin) / ((float) inArgs.nBins);
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
	int volSize[3]; // size of full three dimensional volume [z, x, y]
	int nSubVols[3]; // number of subvolumes in zxy
	int sizeSubVols[3]; // size of each subvolume in zxy (should be uneven)
	int spacingSubVols[3]; // spacing of subvolumes (they can overlap)
	int origin[3]; // position of the very first element [0, 0, 0]
	int end[3]; // terminal value
	int nElements; // total number of elements
	
	// get functions
	void calculate_nsubvols();
	int get_nSubVols(const uint8_t iDim) const {return nSubVols[iDim];};
	int get_nSubVols() const {return nSubVols[0] * nSubVols[1] * nSubVols[2];};
	int get_nElements() const;

};


// calculate number of subvolumes
void gridder::calculate_nsubvols()
{
	// number of subvolumes
	# pragma unroll
	for (unsigned char iDim = 0; iDim < 3; iDim++)
	{
		const int lastIdx = volSize[iDim] - 1;
		nSubVols[iDim] = (lastIdx - origin[iDim]) / spacingSubVols[iDim] + 1;
		origin[iDim] = (sizeSubVols[iDim] - 1) / 2;
		end[iDim] = origin[iDim] + (nSubVols[iDim] - 1) * spacingSubVols[iDim];
	}

	return;
}

// returns number of element in the volume
int gridder::get_nElements() const
{
	return nElements;
}

// small helper class to show errors
class cudaTools
{
	public:
		void checkCudaErr(cudaError_t err, const char* msgErr);
		cudaError_t cErr;
};

void cudaTools::checkCudaErr(cudaError_t err, const char* msgErr)
{
	if (err != cudaSuccess)
	{
		printf("There was some CUDA error: %s, %s\n",
			msgErr, cudaGetErrorString(err));
		throw "CudaError";
	}
	return;
}

// main class used for histogram equ
class histeq: public cudaTools, public gridder
{
	public:
		float* dataMatrix; // 3d matrix containing input volume and maybe output
		
		float* dataOutput; // this will be used to store the output if overwrite is disabled
		bool isDataOutputAlloc = 0;
		
		float* cdf; // contains cummulative distribution function for each subol
		bool isCdfAlloc = 0;

		float* maxValBin; // maximum value occuring in each subvolume [iZ, iX, iY]
		float* minValBin; // maximum value occuring in each subvolume [iZ, iX, iY]
		bool isMaxValBinAlloc = 0;
		
		int nBins = 20; // number of histogram bins
		float noiseLevel = 0.1; // noise level threshold (clipLimit)

		histeq();
		~histeq();
		void calculate_cdf_gpu();
		void equalize_gpu();
};

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
		inArgs.range[iDim] = (int) (sizeSubVols[iDim] - 1) / 2; // range of each bin in each direction
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
	const int nCdf = nBins * nSubVols[0] * nSubVols[1] * nSubVols[2];
	const int nSubs = nSubVols[0] * nSubVols[1] * nSubVols[2];

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
	cErr = cudaMemcpy(dataMatrix, dataMatrix_dev, nElements * sizeof(float), cudaMemcpyDeviceToHost);
	checkCudaErr(cErr, "Problem while copying data matrix back from device");

	cudaFree(dataMatrix_dev);
	cudaFree(cdf_dev);
	cudaFree(maxValBin_dev);
	cudaFree(minValBin_dev);

	return;
}

int main()
{
	// generate input volume matrix and assign random values to it
	const int volSize[3] = {600, 500, 400};
	float* inputVol = new float[volSize[0] * volSize[1] * volSize[2]];
	for(int iIdx = 0; iIdx < (volSize[0] * volSize[1] * volSize[2]); iIdx ++)
		inputVol[iIdx] = ((float) rand()) / ((float) RAND_MAX);

	// initialize some parameters
	histeq histHandler;

	for (uint8_t iDim = 0; iDim < 3; iDim++)
		histHandler.volSize[iDim] = volSize[iDim];
	
	histHandler.dataMatrix = inputVol;
	
	// here goes the actual caluclation
	histHandler.calculate_cdf_gpu();
	histHandler.equalize_gpu();

	delete[] inputVol;
	return 0;
}