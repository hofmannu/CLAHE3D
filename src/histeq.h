/*
	File: histeq.h
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
*/



#include <fstream>
#include <cstdint>
#include <cmath>
#include <chrono> // for timing
#include <thread> // for multithreading
#include <vector>
#include "gridder.h"
#include "vector3.h"

#if USE_CUDA
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include "cudaTools.cuh"
#endif

#ifndef HISTEQSETTINGS_H
#define HISTEQSETTINGS_H

struct histeqsett
{
	std::size_t nBins = 255;
	float noiseLevel = 0.1;
	std::size_t sizeSubVols[3] = {11, 11, 11};
	std::size_t spacingSubVols[3] = {5, 5, 5};
};

#endif

#ifndef HISTEQ_H
#define HISTEQ_H

class histeq: 
#if USE_CUDA
	public cudaTools, 
#endif
public gridder
{

	private:

		// overwrite behaviour
		bool flagOverwrite = 0; // should we put output over input matrix?
		float* dataOutput; // this will be used to store the output if overwrite is disabled
		bool isDataOutputAlloc = 0;

		float* dataMatrix; // 3d matrix containing input volume and maybe output

		float* cdf; // contains cummulative distribution function for each subol
		// structure: iBin + iZ * nBins + iX * nBins * nZ + iY * nBins * nZ * nX 
		bool isCdfAlloc = 0;

		float* maxValBin; // maximum value occuring in each subvolume [iZ, iX, iY]
		float* minValBin; // maximum value occuring in each subvolume [iZ, iX, iY]
		bool isMaxValBinAlloc = 0;

		std::size_t nBins = 255; // number of histogram bins
		float noiseLevel = 0.1f; // noise level threshold (clipLimit)
				
		// float overallMax = 0; // maximum value in entire data matrix

		// private functions
		void calculate_sub_cdf(
			const vector3<std::size_t>& startVec, 
			const vector3<std::size_t>& endVec, 
			const vector3<std::size_t>& iBin,
			float* localData);
	
		float* create_ptrOutput(); // returns a pointer to the output memory bit, careful

		float tCdf = 0; // time required for cdf calculation [ms]
		float tEq = 0; // time required for equilization [ms]
		
		std::size_t nThreads = 1; // number of multiprocessing units
		std::vector<std::thread> workers;

	public:
		// class constructor and destructor
		histeq();
		~histeq();

		// cpu functions
		void calculate_cdf_range(const std::size_t zStart, const std::size_t zStop);
		void calculate_cdf();

		void equalize_range(float* outputArray, const std::size_t idxStart, const std::size_t idxStop);
		void equalize();

		// gpu functions
#if USE_CUDA
		void calculate_cdf_gpu();
		void equalize_gpu();
#endif
	
		float* get_pnoiseLevel() { return &noiseLevel;};
		bool* get_pflagOverwrite() {return &flagOverwrite;};
		std::size_t* get_pnBins() {return &nBins;};

		template<typename T>
		inline float get_icdf(const vector3<T>& iSubVol, const float value) const;
		
		template<typename T>
		float get_icdf(const T ix, const T iy, const T iz, const float value) const;

		// return values from cdf array
		float get_cdf(const std::size_t _iBin, 
			const std::size_t iXSub, const std::size_t iYSub, const std::size_t iZSub) const;
		float get_cdf(const std::size_t iBin, const vector3<std::size_t> iSub) const;
		float get_cdf(const std::size_t iBin, const std::size_t iSubLin) const;
		float get_cdf(const std::size_t linIdx) const {return cdf[linIdx];};
		
		float* get_pcdf() {return cdf;}; // return pointer to cdf
		std::size_t get_ncdf() const {return (nSubVols[0] * nSubVols[1] * nSubVols[2] * nBins);};

		float* get_ptrOutput(); // return pointer to output array

		float get_tCdf() const {return tCdf;};
		float get_tEq() const {return tEq;};
		
		float get_outputValue(const std::size_t iElem) const;
		float get_outputValue(const vector3<std::size_t>& idx) const;
		float get_outputValue(const std::size_t iZ, const std::size_t iX, const std::size_t iY) const;
		float get_minValBin(const std::size_t zBin, const std::size_t xBin, const std::size_t yBin);
		float get_maxValBin(const std::size_t zBin, const std::size_t xBin, const std::size_t yBin);

		// set functions
		void set_nBins(const std::size_t _nBins);

		void set_noiseLevel(const float _noiseLevel);
		void set_data(float* _dataMatrix);
		void set_overwrite(const bool _flagOverwrite);

};


#endif