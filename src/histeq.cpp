#include "histeq.h"

// class constructor
histeq::histeq()
{

}

histeq::~histeq()
{
	if (isCdfAlloc)
		delete[] cdf;

	if (isMaxValBinAlloc)
		delete[] maxValBin;
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

void histeq::calculate()
{
	calculate_nsubvols();
	getOverallMax();
	
	// allocate memory for transfer function
	if (isCdfAlloc)
		delete[] cdf;
	cdf = new float[nBins * nSubVols[0] * nSubVols[1] * nSubVols[2]];
	isCdfAlloc = 1;

	if (isMaxValBinAlloc)
		delete[] maxValBin;
	maxValBin = new float[nSubVols[0] * nSubVols[1] * nSubVols[2]];
	isMaxValBinAlloc = 1;

	uint64_t idxSub[3]; // index of current subvolume
	uint64_t idxStart[3]; // start index of current subvolume
	uint64_t idxEnd[3]; // end index of current subvolume

	// calculate histogram for each individual block
	for(idxSub[2] = 0; idxSub[2] < nSubVols[2]; idxSub[2]++) // for each z subvolume
	{
		for(idxSub[1] = 0; idxSub[1] < nSubVols[1]; idxSub[1]++) // for each y subvolume
		{
			for (idxSub[0] = 0; idxSub[0] < nSubVols[0]; idxSub[0]++) // for each x subvolume
			{
				// get stopping index
				for(unsigned char iDim = 0; iDim < 3; iDim++)
				{
					idxStart[iDim] = idxSub[iDim] *  sizeSubVols[iDim];
					idxEnd[iDim] = idxStart[iDim] + sizeSubVols[iDim] - 1;
					// for last volumes it might occur that we are crossing array
					// boundaries --> reduce
					if (idxEnd[iDim] >= volSize[iDim]){
						idxEnd[iDim] = volSize[iDim] - 1;
					}
				}
					
				getCDF(idxStart[0], idxEnd[0],
					idxStart[0], idxEnd[1],
					idxStart[0], idxEnd[2],
					idxSub[0], idxSub[1], idxSub[2]);
				// invertCDF(idxRun[0], idxRun[1], idxRun[2]);
			}
		}
	}
}

void histeq::getCDF(
	const uint64_t zStart, const uint64_t zEnd, // z start & stop idx
	const uint64_t xStart, const uint64_t xEnd, // x start & stop idx
	const uint64_t yStart, const uint64_t yEnd, // y start & stop idx
	const uint64_t iZ, const uint64_t iX, const uint64_t iY) // bin index
{

	const uint64_t idxSubVol = iZ + iX * nSubVols[0] + iY * nSubVols[0] * nSubVols[1];
	float* hist = new float[nBins]; // histogram of subvolume, only temporarily requried

	// volume is indexed as iz + ix * nz + iy * nx * nz
	// cdf is indexed as [iBin, iZSub, iXSub, iYSub]
	uint64_t yOffset, xOffset;


	// reset bins to zero before summing them up
	for (uint64_t iBin = 0; iBin < nBins; iBin++)
		hist[iBin] = 0;

	// calculate local maximum
	float tempMax = 0; // temporary variable to reduce data access
	for (uint64_t iY = yStart; iY <= yEnd; iY++)
	{
		yOffset = iY * volSize[0] * volSize[1];
		for(uint64_t iX = xStart; iX <= xEnd; iX++)
		{
			xOffset = iX * volSize[0];
			for(uint64_t iZ = zStart; iZ <= zEnd; iZ++)
			{
				const float currVal = dataMatrix[iZ + xOffset + yOffset];
				if (currVal > tempMax)
				{
					tempMax = currVal;
				}
			}
		}
	}
	maxValBin[idxSubVol] = tempMax;

	// calculate size of each bin
	const float binRange = (tempMax - noiseLevel) / ((float) nBins);

	uint64_t validVoxelCounter = 0;
	// sort values into bins which are above clipLimit
	for (uint64_t iY = yStart; iY <= yEnd; iY++)
	{
		yOffset = iY * volSize[0] * volSize[1];
		for(uint64_t iX = xStart; iX <= xEnd; iX++)
		{
			xOffset = iX * volSize[0];
			for(uint64_t iZ = zStart; iZ <= zEnd; iZ++)
			{
				// only add to histogram if above clip limit
				if (dataMatrix[iZ + xOffset + yOffset] >= noiseLevel)
				{
					uint64_t iBin = (dataMatrix[iZ + xOffset + yOffset] - noiseLevel) / binRange;

					// special case for maximum values in subvolume (they gonna end up
					// one index above)
					if (iBin == nBins)
						iBin = nBins - 1;

					hist[iBin] += 1;
					validVoxelCounter++;
				}
			}
		}
	}

	// normalize so that sum of histogram is 1
	for (uint64_t iBin = 0; iBin < nBins; iBin++)
		hist[iBin] /= (float) (validVoxelCounter);
	
	// calculate cummulative sum and scale along y
	float cdfTemp = 0;
	uint64_t binOffset = nBins * idxSubVol;
	for (uint64_t iBin = 0; iBin < nBins; iBin++)
	{
		cdfTemp += hist[iBin];
		cdf[binOffset + iBin] = cdfTemp;
	}

	// free hist memory
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
	if (value < noiseLevel)
	{
		return 0;
	}
	else
	{
		// get index describes the 3d index of the subvolume
		const uint64_t subVolIdx = iZ + nSubVols[0] * iX + nSubVols[0] * nSubVols[1] * iY;
		const uint64_t subVolOffset = nBins * subVolIdx;
		const float vInterp = (value - noiseLevel) / maxValBin[subVolIdx]; // should now span 0 to 1 
		
		// it can happen that the voxel value is higher then the max value detected
		// in the next volume. In this case we crop it to the maximum permittable value
		const uint64_t binOffset = (vInterp > 1) ? 
			nBins - 1 + subVolOffset
			: (vInterp * ((float) nBins - 1.0) + 0.5) + subVolOffset;

		return cdf[binOffset];
	}
	
}

void histeq::calculate_nsubvols(){
	// number of subvolumes
	for (unsigned char iDim = 0; iDim < 3; iDim++)
		nSubVols[iDim] = (volSize[iDim] - 2) / sizeSubVols[iDim] + 1;
	
	printf("[histeq] number of subvolumes: %ld, %ld, %ld\n", 
			nSubVols[0], nSubVols[1], nSubVols[2]);

	return;
}

// define number of bins during eq
void histeq::setNBins(const uint64_t _nBins){
	nBins = _nBins;
	return;
}

// define noiselevel of dataset as minimum occuring value
void histeq::setNoiseLevel(const float _noiseLevel){
	noiseLevel = _noiseLevel;
	return;
}

// size of full three dimensional volume
void histeq::setVolSize(const uint64_t* _volSize){
	for(unsigned char iDim = 0; iDim < 3; iDim++)
		volSize[iDim] = _volSize[iDim];

	nElements = volSize[0] * volSize[1] * volSize[2];
	return;
}

void histeq::setData(float* _dataMatrix){
	dataMatrix = _dataMatrix;
	return;
}

void histeq::setSizeSubVols(const uint64_t* _subVolSize){
	for(unsigned char iDim = 0; iDim < 3; iDim++)
		sizeSubVols[iDim] = _subVolSize[iDim];

	return;
}
