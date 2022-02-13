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
#include "gridder.h"

#if USE_CUDA
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include "cudaTools.cuh"
#endif

class histeq: 
#if USE_CUDA
	public cudaTools, 
#endif
public gridder
{

	private:

		// overwrite behaviour
		bool flagOverwrite = 1; // should we put output over input matrix?
		float* dataOutput; // this will be used to store the output if overwrite is disabled
		bool isDataOutputAlloc = 0;

		float* dataMatrix; // 3d matrix containing input volume and maybe output

		float* cdf; // contains cummulative distribution function for each subol
		// structure: iBin + iZ * nBins + iX * nBins * nZ + iY * nBins * nZ * nX 
		bool isCdfAlloc = 0;

		float* maxValBin; // maximum value occuring in each subvolume [iZ, iX, iY]
		float* minValBin; // maximum value occuring in each subvolume [iZ, iX, iY]
		bool isMaxValBinAlloc = 0;

		uint64_t nBins; // number of histogram bins
		float noiseLevel; // noise level threshold (clipLimit)
				
		float overallMax = 0; // maximum value in entire data matrix

		// private functions
		void calculate_sub_cdf(
			const uint64_t zStart, const uint64_t zEnd,
			const uint64_t xStart, const uint64_t xEnd,
			const uint64_t yStart, const uint64_t yEnd, 
			const uint64_t iZ, const uint64_t iX, const uint64_t iY);
	
		float* create_ptrOutput(); // returns a pointer to the output memory bit, careful

	public:
		// class constructor and destructor
		histeq();
		~histeq();

		// cpu functions
		void calculate_cdf();
		void equalize();

		// gpu functions
#if USE_CUDA
		void calculate_cdf_gpu();
		void equalize_gpu();
#endif

		float get_icdf(
			const uint64_t iZ, const uint64_t iX, const uint64_t iY, const float value);

		float get_cdf( // return a specific position of our cdf array
			const uint64_t _iBin, 
			const uint64_t iZSub, const uint64_t iXSub, const uint64_t iYSub);
		float get_cdf(const uint64_t linIdx) const {return cdf[linIdx];};
		float* get_pcdf() {return cdf;}; // return pointer to cdf
		uint64_t get_ncdf() const {return (nSubVols[0] * nSubVols[1] * nSubVols[2] * nBins);};

		float* get_ptrOutput(); // return pointer to output array
		float get_outputValue(const uint64_t iElem) const;
		float get_outputValue(const uint64_t iZ, const uint64_t iX, const uint64_t iY) const;
		float get_minValBin(const uint64_t zBin, const uint64_t xBin, const uint64_t yBin);
		float get_maxValBin(const uint64_t zBin, const uint64_t xBin, const uint64_t yBin);

		// set functions
		void set_nBins(const uint64_t _nBins);
		void set_noiseLevel(const float _noiseLevel);
		void set_data(float* _dataMatrix);
		void set_overwrite(const bool _flagOverwrite);

};


#endif