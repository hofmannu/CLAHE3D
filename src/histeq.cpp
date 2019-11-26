#include "histeq.h"

void histeq::calculate(){
	calculate_nsubvols();

	// allocate memory for transfer function
	cdf = new float[nBins * nSubVols[0] * nSubVols[1] * nSubVols[2]];
	icdf = new float[nBins * nSubVols[0] * nSubVols[1] * nSubVols[2]];
	maxVal = new float[nSubVols[0] * nSubVols[1] * nSubVols[2]];

	// calculate histogram for each individual block
	for(uint64_t iY = 0; iY < nSubVols[2]; iY++){
		for(uint64_t iX = 0; iX < nSubVols[1]; iX++){
			for (uint64_t iZ = 0; iZ < nSubVols[0]; iZ++){
				getCDF(iZ * sizeSubVols[0], (iZ + 1) * sizeSubVols[0] - 1,
					iX * sizeSubVols[1], (iX + 1) * sizeSubVols[1] - 1,
					iY * sizeSubVols[2], (iY + 1) * sizeSubVols[2] - 1,
					iZ, iX, iY);
				invertCDF(iZ, iX, iY);
			}
		}
	}
}

void histeq::invertCDF(uint64_t iZ, uint64_t iX, uint64_t iY){
		
	// until now our cdf ranges from 0 to 1
	// 0 represents noiseLevel / clipLimit
	// 1 represents local max value stored in maxVal[iZ, iX, iY]
	// we now want to subdivide along y in nBins sublevels
	
	uint64_t binOffset = nBins * (iZ + iX * nSubVols[0] +	iY * nSubVols[0] * nSubVols[1]);

	float ratio = 0; // runs from 0 to almost 1
	uint64_t cdfBin = 0; // starting bin for interpolation
	float currDelta = 0;
	for(uint64_t icdfBin = 0; icdfBin < nBins; icdfBin++){
		ratio = (float) icdfBin / (float) nBins;
		
		// search for position in cdf which is just above ratio
		// we can do this because cdf is strictly rising from 0 to 1
		while((cdf[cdfBin + binOffset] < ratio) && (cdfBin < (nBins - 1))){
			cdfBin++;
			currDelta = cdf[cdfBin + binOffset] - cdf[cdfBin + binOffset - 1];
		}
	
		if (cdfBin == (nBins - 1))
			icdf[icdfBin + binOffset] = nBins - 1;
		else if (cdfBin == 0)
			icdf[icdfBin + binOffset] = 0;
		else
			icdf[icdfBin + binOffset] = cdfBin - (ratio - cdf[cdfBin + binOffset - 1]) / currDelta; 
	
		printf("%f, ", icdf[icdfBin + binOffset]);	
	}
	printf("\n");
	// output for deebugging purpose only
	// printf("runs: %ld, %ld, %ld\n", iZ, iX, iY);
	return;
}


void histeq::getCDF(
	const uint64_t zStart, const uint64_t zEnd, // z start & stop idx
	const uint64_t xStart, const uint64_t xEnd, // x start & stop idx
	const uint64_t yStart, const uint64_t yEnd, // y start & stop idx
	const uint64_t iZ, const uint64_t iX, uint64_t iY){

	uint64_t subVolOffset = iZ + iX * nSubVols[0] + iY * nSubVols[0] * nSubVols[1];
	float* hist = new float[nBins]; // histogram of subvolume, only temporarily requried

	// volume is indexed as iz + ix * nz + iy * nx * nz
	// cdf is indexed as [iBin, iZSub, iXSub, iYSub]
	uint64_t yOffset, xOffset;

	// find maximum value in volume
	float maxValTemp = noiseLevel; 
	for (uint64_t iY = yStart; iY <= yEnd; iY++){
		yOffset = iY * volSize[0] * volSize[1]; // x offset in data matrix
		for(uint64_t iX = xStart; iX <= xEnd; iX++){
			xOffset = iX * volSize[0]; // y offset in data matrix
			for(uint64_t iZ = zStart; iZ <= zEnd; iZ++){
				if (maxValTemp < dataMatrix[iZ + xOffset + yOffset])
					maxValTemp = dataMatrix[iZ + xOffset + yOffset];
			}
		}
	}
	
	// store maxVal in matrix for each bin
	maxVal[subVolOffset] = maxValTemp;
	
	// calculate size of each bin
	float binRange = (maxValTemp - noiseLevel) / ((float) nBins);

	// reset bins to zero before summing them up
	for (uint64_t iBin = 0; iBin < nBins; iBin++)
		hist[iBin] = 0;

	uint64_t validVoxelCounter = 0;
	// sort values into bins which are above clipLimit
	for (uint64_t iY = yStart; iY <= yEnd; iY++){
		yOffset = iY * volSize[0] * volSize[1];
		for(uint64_t iX = xStart; iX <= xEnd; iX++){
			xOffset = iX * volSize[0];
			for(uint64_t iZ = zStart; iZ <= zEnd; iZ++){
				// only add to histogram if above clip limit
				if (dataMatrix[iZ + xOffset + yOffset] >= noiseLevel){
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
	uint64_t binOffset = nBins * subVolOffset;
	for (uint64_t iBin = 0; iBin < nBins; iBin++){
		printf("%f, ", hist[iBin]);
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
	const float value){ // value to extract
	// idx describes the 3d index of the subvolume
	
	// convert value to somethinng which ranges from 0 to 1
	uint64_t subVolOffset = iZ + nSubVols[0] * iX + nSubVols[0] * nSubVols[1] * iY;
	float vInterp = maxVal[subVolOffset]; // get maximum value of offset
	vInterp = (value - noiseLevel) / vInterp; // should now span 0 to 1 
	
	if (vInterp < 0){ // if we are already below noiseLevel send back a 0
		vInterp = 0;
	}else{
		// it can happen that the voxel value is higher then the max value detected
		// in the next volume. In this case we crop it to the maximum permittable value
		if (vInterp > 1)
			vInterp = 1;

		// convert range 0 ... 1 to index in icdf
		vInterp = vInterp * (float) nBins + 0.5;
		uint64_t binOffset = (uint64_t) vInterp + nBins * subVolOffset;
		vInterp = icdf[binOffset];
	}
	
	return vInterp;
}

void histeq::calculate_nsubvols(){
	for (unsigned char iDim = 0; iDim < 3; iDim++){
		// number of fully fitting subvolumes
		nSubVols[iDim] = volSize[iDim] / sizeSubVols[iDim];
		// if there is an overhang, add another volume
		if (nSubVols[iDim] % sizeSubVols[iDim])
			nSubVols[iDim]++;
	}
	printf("[histeq] number of subvolumes: %ld, %ld, %ld\n", 
			nSubVols[0], nSubVols[1], nSubVols[2]);

	return;
}

void histeq::setNBins(const uint64_t _nBins){
	nBins = _nBins;
	return;
}

void histeq::setNoiseLevel(const float _noiseLevel){
	noiseLevel = _noiseLevel;
	return;
}

void histeq::setVolSize(const uint64_t* _volSize){
	for(unsigned char iDim = 0; iDim < 3; iDim++)
		volSize[iDim] = _volSize[iDim];

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
