#include "vector3.h"
#include <iostream>

using namespace std;

#ifndef GENFILT_H
#define GENFILT_H

class genfilt
{
private:
	vector3<int> dataSize;
	vector3<int> kernelSize;
	vector3<int> paddedSize;
	vector3<int> range;
	float* dataInput;

	// padded version of input data
	float* dataPadded;
	bool isDataPaddedAlloc = 0;
	
	float* dataOutput; // cannot be same as dataInput
	bool isDataOutputAlloc = 0;

	float* kernel;

public:
	~genfilt();

	// execution
	void alloc_output();
	void alloc_padded();
	void padd();
	void conv();
	void conv_gpu();

	// set functions
	void set_dataInput(float* _dataInput);
	void set_kernel(float* _kernel);
	void set_dataSize(const vector3<int> _dataSize);
	void set_kernelSize(const vector3<int> _kernelSize);

	int* get_pkernelSize() {return &kernelSize.x;};
	vector3<int> get_kernelSize() const {return kernelSize;};

	// get functions
	vector3<int> get_range() const {return range;};
	vector3<int> get_paddedSize() const {return (dataSize + range * 2);};
	int get_nKernel() const {return kernelSize.elementMult();};
	int get_nData() const {return dataSize.elementMult();};
	int get_nPadded() const {return paddedSize.elementMult();};
	float* get_pdataOutput() {return dataOutput;};

	int get_dataDim(const uint8_t iDim) const {return dataSize[iDim];};
};

#endif