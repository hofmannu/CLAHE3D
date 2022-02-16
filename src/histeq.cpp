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

	// this memory set is much slower then the one in the kernel
	// cErr = cudaMemset(cdf_dev, 0, nCdf * sizeof(float));
	// checkCudaErr(cErr, "Could not initialize CDF to zeros");

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

	// calculate histogram for each individual block
	for (int iZSub = 0; iZSub < nSubVols.z; iZSub++) // for each x subvolume
	{
		for(int iYSub = 0; iYSub < nSubVols.y; iYSub++) // for each z subvolume
		{
			for(int iXSub = 0; iXSub < nSubVols.x; iXSub++) // for each y subvolume
			{
				const vector3<int> idxSub = {iXSub, iYSub, iZSub}; // index of current subvolume
				const vector3<int> idxStart = get_startIdxSubVol(idxSub);
				const vector3<int> idxEnd = get_stopIdxSubVol(idxSub);
				// printf("subvol ranges from %d, %d, %d to %d, %d, %d\n",
				// 	idxStart.x, idxStart.y, idxStart.z, idxEnd.x, idxEnd.y, idxEnd.z);
				calculate_sub_cdf(idxStart, idxEnd, idxSub);
			}
		}
	}
	return;
}

// get cummulative distribution function for a certain bin 
void histeq::calculate_sub_cdf(
	const vector3<int> startVec, const vector3<int> endVec, const vector3<int> iBin) // bin index
{

	const int idxSubVol = iBin.x + nSubVols.x * (iBin.y + iBin.z * nSubVols.y);
	float* localCdf = &cdf[idxSubVol * nBins]; // histogram of subvolume, only temporarily requried

	// volume is indexed as iz + ix * nz + iy * nx * nz
	// cdf is indexed as [iBin, iZSub, iXSub, iYSub]

	// reset bins to zero before summing them up
	for (int iBin = 0; iBin < nBins; iBin++)
		localCdf[iBin] = 0;

	// calculate local maximum and minimum
	const float firstVal = dataMatrix[
		startVec.x + volSize.x * (startVec.y + volSize.y * startVec.z)];
	float tempMax = firstVal; // temporary variable to reduce data access
	float tempMin = firstVal;
	for (int iZ = startVec.z; iZ <= endVec.z; iZ++)
	{
		const int zOffset = iZ * volSize.x * volSize.y;
		for(int iY = startVec.y; iY <= endVec.y ; iY++)
		{
			const int yOffset = iY * volSize.x;
			for(int iX = startVec.x; iX <= endVec.x; iX++)
			{
				const float currVal = dataMatrix[iX + yOffset + zOffset];
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
	for (int iZ = startVec.z; iZ <= endVec.z; iZ++)
	{
		const int zOffset = iZ * volSize.x * volSize.y;
		for(int iY = startVec.y; iY <= endVec.y; iY++)
		{
			const int yOffset = iY * volSize.x;
			for(int iX = startVec.x; iX <= endVec.x; iX++)
			{
				const float currVal = dataMatrix[iX + yOffset + zOffset];
				// only add to histogram if above clip limit
				if (currVal >= noiseLevel)
				{
					const int iBin = (currVal - tempMin) / binRange;
					// special case for maximum values in subvolume (they gonna end up
					// one index above)

					if (iBin >= nBins)
					{
						localCdf[nBins - 1] += 1;
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
	for (int iBin = 0; iBin < nBins; iBin++)
	{
		cdfTemp += localCdf[iBin];
		localCdf[iBin] = cdfTemp - zeroElem;
	}

	// now we scale cdf to max == 1 (first value is 0 anyway)
	const float cdfMax = localCdf[nBins - 1];
	if (cdfMax > 0)
	{
		for (int iBin = 1; iBin < nBins; iBin++)
		{
			localCdf[iBin] /= cdfMax;
		}
	}
	else
	{
		for (int iBin = 1; iBin < nBins; iBin++)
		{
			localCdf[iBin] = ((float) iBin) / ((float) nBins - 1);
		}
	}

	return;
}

// returns the inverted cummulative distribution function
float histeq::get_icdf(const vector3<int> iSubVol, const float currValue) // value to extract
{
	// if we are below noise level, directy return
	const int subVolIdx = iSubVol.x + nSubVols.x * (iSubVol.y + nSubVols.y * iSubVol.z);
	if (currValue <= minValBin[subVolIdx])
	{
		return 0.0;
	}
	else
	{
		// get index describes the 3d index of the subvolume
		const int subVolOffset = nBins * subVolIdx;
		const float vInterp = (currValue - minValBin[subVolIdx]) / 
			(maxValBin[subVolIdx] - minValBin[subVolIdx]); // should now span 0 to 1 		
		
		// it can happen that the voxel value is higher then the max value detected
		// in the next volume. In this case we crop it to the maximum permittable value
		const int binOffset = (vInterp > 1.0) ? 
			(nBins - 1 + subVolOffset)
			: fmaf(vInterp, (float) nBins - 1.0, 0.5) + subVolOffset;


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
	eq_arguments inArgs;
	#pragma unroll
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		inArgs.volSize[iDim] = volSize[iDim];
		inArgs.origin[iDim] = origin[iDim];
		inArgs.end[iDim] = endIdx[iDim];
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

	int neighbours[6]; // index of next neighbouring elements
	float ratio[3]; // ratios in z x y
	for(int iZ = 0; iZ < volSize.z; iZ++)
	{
		for (int iY = 0; iY < volSize.y; iY++)
		{
			for (int iX = 0; iX < volSize.x; iX++)
			{
				const int idxVolLin = iX + volSize.x * (iY + volSize.y * iZ);
				const float currValue = dataMatrix[idxVolLin];
	
				const vector3<int> position = {iX, iY, iZ};
				get_neighbours(position, neighbours, ratio);
				// printf("%d, %d, %d\n", neighbours[0], neighbours[2], neighbours[4]);

				// get values from all eight corners
				const float value[8] = {
					get_icdf({neighbours[0], neighbours[2], neighbours[4]}, currValue),
					get_icdf({neighbours[0], neighbours[2], neighbours[5]}, currValue),
					get_icdf({neighbours[0], neighbours[3], neighbours[4]}, currValue),
					get_icdf({neighbours[0], neighbours[3], neighbours[5]}, currValue),
					get_icdf({neighbours[1], neighbours[2], neighbours[4]}, currValue),
					get_icdf({neighbours[1], neighbours[2], neighbours[5]}, currValue),
					get_icdf({neighbours[1], neighbours[3], neighbours[4]}, currValue),
					get_icdf({neighbours[1], neighbours[3], neighbours[5]}, currValue)};

				// trilinear interpolation
				ptrOutput[idxVolLin] =
					(1 - ratio[0]) * (
						(1 - ratio[1]) * (
							value[0] * (1 - ratio[2]) +
							value[1] * ratio[2]
						) + ratio[1] * (
							value[2] * (1 - ratio[2]) +
							value[3] * ratio[2] 
						)
					) + ratio[0] * (
						(1 - ratio[1]) * (
							value[4] * (1 - ratio[2]) +
							value[5] * ratio[2]
						) + ratio[1] * (
							value[6] * (1 - ratio[2]) +
							value[7] * ratio[2]
						)
					);

			}
		}
	}
	return;
}



// returns a single value from our cdf function
float histeq::get_cdf(const int iBin, const int iXSub, const int iYSub, const int iZSub) const
{
	const int idx = iBin + nBins * (iXSub + nSubVols[0] * (iYSub +  nSubVols[1] * iZSub));
	return cdf[idx];
}

float histeq::get_cdf(const int iBin, const vector3<int> iSub) const
{
	const int idx = iBin + nBins * (iSub.x + nSubVols[0] * (iSub.y +  nSubVols[1] * iSub.z));
	return cdf[idx];
}

float histeq::get_cdf(const int iBin, const int iSubLin) const
{
	const int idx = iBin + nBins * iSubLin;
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
float histeq::get_outputValue(const int iElem) const
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
float histeq::get_outputValue(const vector3<int> idx) const
{
	const int linIdx = idx.x + volSize.x * (idx.y + volSize.y * idx.z);
	return get_outputValue(linIdx);
} 

// returns output value for 3d index
float histeq::get_outputValue(const int iZ, const int iX, const int iY) const
{
	const int linIdx = iZ + volSize[0] * (iX + volSize[1] * iY);
	return get_outputValue(linIdx);
} 

// returns the minimum value of a bin
float histeq::get_minValBin(const int zBin, const int xBin, const int yBin)
{
	const int idxBin = zBin + nSubVols[0] * (xBin + nSubVols[1] * yBin);
	return minValBin[idxBin];
}

// returns the maximum value of a bin
float histeq::get_maxValBin(const int zBin, const int xBin, const int yBin)
{
	const int idxBin = zBin + nSubVols[0] * (xBin + nSubVols[1] * yBin);
	return maxValBin[idxBin];
}

// define number of bins during eq
void histeq::set_nBins(const int _nBins)
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