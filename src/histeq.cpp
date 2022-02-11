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

uint64_t histeq::getStartIdxSubVol(const uint64_t iSub, const uint8_t iDim)
{
	const int64_t centerPos = (int64_t) iSub * spacingSubVols[iDim];
	int64_t startIdx = centerPos - ((int) sizeSubVols[iDim] - 1) / 2; 
	startIdx = (startIdx < 0) ? 0 : startIdx;
	return (uint64_t) startIdx;
}

uint64_t histeq::getStopIdxSubVol(const uint64_t iSub, const uint8_t iDim)
{
	const int64_t centerPos = (int64_t) iSub * spacingSubVols[iDim];
	int64_t stopIdx = centerPos + ((int) sizeSubVols[iDim] - 1) / 2; 
	stopIdx = (stopIdx >= volSize[iDim]) ? (volSize[iDim] - 1) : stopIdx;
	return (uint64_t) stopIdx;
}

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

				// printf("z range: %d ... %d, x range %d ... %d, y range %d ... %d\n",
				// 	(int) idxStart[0], (int) idxEnd[0],
				// 	(int) idxStart[1], (int) idxEnd[1],
				// 	(int) idxStart[2], (int) idxEnd[2]);
					
				getCDF(idxStart[0], idxEnd[0], // zStart, zEnd
					idxStart[1], idxEnd[1], // xStart, xEnd
					idxStart[2], idxEnd[2], // yStart, yEnd
					idxSub[0], idxSub[1], idxSub[2]);
			}
		}
	}
}

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

void histeq::equalize()
{
	uint64_t neighbours[6]; // index of next neighbouring elements
	float ratio[3]; // ratios in z x y
	float currValue; // value of position in input volume
	for(uint64_t iY = 0; iY < volSize[2]; iY++){
		for (uint64_t iX = 0; iX < volSize[1]; iX++){
			for (uint64_t iZ = 0; iZ < volSize[0]; iZ++){
				currValue = dataMatrix[iZ + volSize[0] * (iX + volSize[1] * iY)];
	
				const uint64_t position[3] = {iZ, iX, iY};
				histGrid.getNeighbours(position, neighbours, ratio);
				
				// assign new value based on trilinear interpolation
				dataMatrix[iZ + volSize[0] * (iX + volSize[1] * iY)] =
				// first two opposing z corners
				((get_icdf(neighbours[0], neighbours[2], neighbours[4], currValue) * (1 - ratio[0]) + 
				get_icdf(neighbours[1], neighbours[2], neighbours[4], currValue) * ratio[0]) 
					* (1 - ratio[1]) +
				// fourth two opposing z corners
				(get_icdf(neighbours[0], neighbours[3], neighbours[5], currValue) * (1 - ratio[0]) + 
				get_icdf(neighbours[1], neighbours[3], neighbours[5], currValue) * ratio[0])
					* ratio[1]) * (1 - ratio[2]) +
				// second two opposing z corners
				((get_icdf(neighbours[0], neighbours[3], neighbours[4], currValue) * (1 - ratio[0]) +
				get_icdf(neighbours[1], neighbours[3], neighbours[4], currValue) * ratio[0])
					* (1 - ratio[1]) +
				// third two opposing z corners
				(get_icdf(neighbours[0], neighbours[2], neighbours[5], currValue) * (1 - ratio[0]) +
				get_icdf(neighbours[1], neighbours[2], neighbours[5], currValue) * ratio[0])
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
