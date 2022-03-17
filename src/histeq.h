/*
	File: histeq.h
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
*/

#ifndef HISTEQSETTINGS_H
#define HISTEQSETTINGS_H

struct histeqsett
{
	int nBins = 255;
	float noiseLevel = 0.1;
	int sizeSubVols[3] = {11, 11, 11};
	int spacingSubVols[3] = {5, 5, 5};
};

#endif

#ifndef HISTEQ_H
#define HISTEQ_H

#include <fstream>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <thread>
#include "gridder.h"
#include "vector3.h"

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

		int nBins = 255; // number of histogram bins
		float noiseLevel = 0.1f; // noise level threshold (clipLimit)
				
		// float overallMax = 0; // maximum value in entire data matrix

		// private functions
		void calculate_sub_cdf(
			const vector3<int> startVec, 
			const vector3<int> endVec, 
			const vector3<int> iBin);
	
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
	
		float* get_pnoiseLevel() { return &noiseLevel;};
		bool* get_pflagOverwrite() {return &flagOverwrite;};
		int* get_pnBins() {return &nBins;};

		float get_icdf(const vector3<int> iSubVol, const float value);

		// return values from cdf array
		float get_cdf(const int _iBin, 
			const int iXSub, const int iYSub, const int iZSub) const;
		float get_cdf(const int iBin, const vector3<int> iSub) const;
		float get_cdf(const int iBin, const int iSubLin) const;
		float get_cdf(const int linIdx) const {return cdf[linIdx];};
		
		float* get_pcdf() {return cdf;}; // return pointer to cdf
		int get_ncdf() const {return (nSubVols[0] * nSubVols[1] * nSubVols[2] * nBins);};

		float* get_ptrOutput(); // return pointer to output array
		
		float get_outputValue(const int iElem) const;
		float get_outputValue(const vector3<int> idx) const;
		float get_outputValue(const int iZ, const int iX, const int iY) const;

		float get_minValBin(const int zBin, const int xBin, const int yBin);
		float get_maxValBin(const int zBin, const int xBin, const int yBin);

		// set functions
		void set_nBins(const int _nBins);
		void set_noiseLevel(const float _noiseLevel);
		void set_data(float* _dataMatrix);
		void set_overwrite(const bool _flagOverwrite);

};


#endif