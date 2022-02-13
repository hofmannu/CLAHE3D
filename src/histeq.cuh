/*
	File: histeq.h
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
*/

#ifndef HISTEQ_H
#define HISTEQ_H

#include <fstream>
#include <cstdint>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

#include "interpGrid.h"
#include "cudaTools.cuh"

class histeq: public cudaTools 
{

	private:
		interpGrid histGrid;

		// private variables
		uint64_t nBins; // number of histogram bins
		float noiseLevel; // noise level threshold (clipLimit)

		uint64_t volSize[3]; // size of full three dimensional volume [z, x, y]
		uint64_t nElements; // number of elements 

		float* dataMatrix; // 3d matrix containing all values
		uint64_t nSubVols[3]; // number of subvolumes in zxy
		uint64_t sizeSubVols[3]; // size of each subvolume in zxy (should be uneven)
		uint64_t spacingSubVols[3]; // spacing of subvolumes (they can overlap)
		
		float* cdf; // contains cummulative distribution function for each subol
		bool isCdfAlloc = 0;
		// structure: iBin + iZ * nBins + iX * nBins * nZ + iY * nBins * nZ * nX 
		
		float* icdf; // inverted version of cummulative distribution function
		
		float* maxValBin; // maximum value occuring in each subvolume [iZ, iX, iY]
		float* minValBin; // maximum value occuring in each subvolume [iZ, iX, iY]
		bool isMaxValBinAlloc = 0;
		float overallMax = 0; // maximum value in entire data matrix

		// private functions
		void calculate_nsubvols();
		void getCDF(
			const uint64_t zStart, const uint64_t zEnd,
			const uint64_t xStart, const uint64_t xEnd,
			const uint64_t yStart, const uint64_t yEnd, 
			const uint64_t iZ, const uint64_t iX, const uint64_t iY);
		// void invertCDF(const uint64_t iZ, uint64_t iX, uint64_t iY);
		void getOverallMax();
		inline uint64_t getStartIdxSubVol(const uint64_t iSub, const uint8_t iDim);
		inline uint64_t getStopIdxSubVol(const uint64_t iSub, const uint8_t iDim);

	public:
		// class constructor and destructor
		histeq();
		~histeq();

		void calculate();
		void equalize();

		// gpu functions
		void calculate_gpu();
		void equalize_gpu();

		float get_icdf(
			const uint64_t iZ, const uint64_t iX, const uint64_t iY, const float value);

		// returns a single value of our cdf array
		float get_cdf(const uint64_t _iBIn, const uint64_t iZSub, const uint64_t iXSub, const uint64_t iYSub);

		void setNBins(const uint64_t _nBins);
		void setNoiseLevel(const float _noiseLevel);
		void setVolSize(const uint64_t* _volSize);
		void setData(float* _dataMatrix);
		void setSizeSubVols(const uint64_t* _subVolSize);
		void setSpacingSubVols(const uint64_t* _spacingSubVols);

};


#endif