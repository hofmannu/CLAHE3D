/*
	a class to run a volumetric kernel over a volume
	Author : Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 
*/

#include "vector3.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>

using namespace std;

#ifndef GENFILT_H
#define GENFILT_H

class genfilt
{
protected:
	vector3<std::size_t> dataSize;
	vector3<std::size_t> kernelSize;
	// vector3<int> kernelSizeArray;
	vector3<std::size_t> paddedSize;
	vector3<std::size_t> range;
	float* dataInput;

	// stuff required for multicore
	std::size_t nThreads = 1;
	vector<std::size_t> zStart;
	vector<std::size_t> zStop;

	// padded version of input data
	float* dataPadded;
	bool isDataPaddedAlloc = 0;
	
	float* dataOutput; // cannot be same as dataInput
	bool isDataOutputAlloc = 0;

	float* kernel;
	float tExec; // execution time in ms

	void conv_range(const std::size_t iRange);

public:
	genfilt();
	~genfilt();

	// execution
	void alloc_output();
	void alloc_padded();
	void padd();
	void conv();
#if USE_CUDA
	void conv_gpu();
#endif

	// set functions
	void set_dataInput(float* _dataInput);
	void set_kernel(float* _kernel);
	void set_dataSize(const vector3<std::size_t> _dataSize);
	void set_kernelSize(const vector3<std::size_t> _kernelSize);

	std::size_t* get_pkernelSize() {return &kernelSize.x;};
	vector3<std::size_t> get_kernelSize() const {return kernelSize;};

	// get functions
	vector3<std::size_t> get_range() const {return range;};
	vector3<std::size_t> get_paddedSize() const {return (dataSize + range * 2);};
	std::size_t get_nKernel() const {return kernelSize.elementMult();};
	std::size_t get_nData() const {return dataSize.elementMult();};
	std::size_t get_nPadded() const {return paddedSize.elementMult();};
	float* get_pdataOutput() {return dataOutput;};
	float get_tExec() const {return tExec;};
	std::size_t get_dataDim(const uint8_t iDim) const {return dataSize[iDim];};
};

#endif