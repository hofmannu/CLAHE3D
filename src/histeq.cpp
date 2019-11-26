#include "histeq.h"

void histeq::getOverallMax(){
	
	overallMax = dataMatrix[0];
	for(uint64_t idx = 1; idx < (volSize[0] * volSize[1] * volSize[2]); idx++){
		if(dataMatrix[idx] > overallMax)
			overallMax = dataMatrix[idx];
	}
	
	return;
}

void histeq::calculate(){
	calculate_nsubvols();
	getOverallMax();
	
	// allocate memory for transfer function
	cdf = new float[nBins * nSubVols[0] * nSubVols[1] * nSubVols[2]];

	uint64_t idxEnd[3];
	uint64_t idxRun[3];
	// calculate histogram for each individual block
	for(idxRun[2] = 0; idxRun[2] < nSubVols[2]; idxRun[2]++){
		for(idxRun[1] = 0; idxRun[1] < nSubVols[1]; idxRun[1]++){
			for (idxRun[0] = 0; idxRun[0] < nSubVols[0]; idxRun[0]++){
				
				// get stopping index
				for(unsigned char iDim = 0; iDim < 3; iDim++){
					idxEnd[iDim] = (idxRun[iDim] + 1) * sizeSubVols[iDim] - 1;
					// for last volumes it might occur that we are crossing array
					// boundaries --> reduce
					if (idxEnd[iDim] >= volSize[iDim]){
						idxEnd[iDim] = volSize[iDim] - 1;
					}
				}
				
				getCDF(idxRun[0] * sizeSubVols[0], idxEnd[0],
					idxRun[1] * sizeSubVols[1], idxEnd[1],
					idxRun[2] * sizeSubVols[2], idxEnd[2],
					idxRun[0], idxRun[1], idxRun[2]);
				// invertCDF(idxRun[0], idxRun[1], idxRun[2]);
			}
		}
	}
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

	// calculate size of each bin
	float binRange = (overallMax - noiseLevel) / ((float) nBins);

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
	
	// get index describes the 3d index of the subvolume
	uint64_t subVolOffset = iZ + nSubVols[0] * iX + nSubVols[0] * nSubVols[1] * iY;
	// convert value to somethinng which ranges from 0 to 1
	float vInterp = (value - noiseLevel) / overallMax; // should now span 0 to 1 
	
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
		vInterp = cdf[binOffset];
	}
	
	return vInterp;
}

void histeq::calculate_nsubvols(){
	// number of subvolumes
	for (unsigned char iDim = 0; iDim < 3; iDim++)
		nSubVols[iDim] = (volSize[iDim] - 2) / sizeSubVols[iDim] + 1;
	
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

// size of full three dimensional volume
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
