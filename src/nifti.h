#ifndef NIFTI_H
#define NIFTI_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <fstream>
#include "../lib/nifti/niftilib/nifti1.h"
#include "vector3.h"

using namespace std;

class nifti
{
private:
  nifti_1_header hdr;
  string filePath;
  float* dataMatrix;
  bool isDataMatrixAlloc = 0;

  void alloc_mem();
  float minVal = 0;
  float maxVal = 0;
public: 

	~nifti();
	int read();
	int read(const string _filePath);

	void save(const string _filePath);

	void print_header();

	nifti_1_header get_header() const {return hdr;};
	void set_header(const nifti_1_header _hdr) {hdr = _hdr; return;};

	// get functions to quickly access properties
	int get_nElements() const {return hdr.dim[1] * hdr.dim[2] * hdr.dim[3];};
	int get_dim(const uint8_t iDim) const {return hdr.dim[iDim+1];};
	float get_res(const uint8_t iDim) const {return hdr.pixdim[iDim+1];};
	vector3<int> get_dim() {return {hdr.dim[1], hdr.dim[2], hdr.dim[3]};};
	const char* get_filePath() const {return filePath.c_str();}; 
	
	float* get_pdataMatrix() {return dataMatrix;};
	float* get_pdataMatrix(const int index) {return &dataMatrix[index];};
	void set_dataMatrix(float* _dataMatrix) {dataMatrix = _dataMatrix; return;};

	float get_min() const {return minVal;};
	float get_max() const {return maxVal;};

	float get_val(const vector3<int> pos) const;

	float get_length(const uint8_t iDim) const {return ((float) hdr.dim[iDim+1] ) * hdr.pixdim[iDim+1];};

	void set_dataMatrix(const int index, const float value) {dataMatrix[index] = value; return;} 
};

#endif