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
	int read_header(const string _filePath);
	void print_header();

	int read_data();
	int read_data(const string _filePath);

	// get functions to quickly access properties
	int get_dim(const uint8_t iDim) {return hdr.dim[iDim+1];};
	vector3<int> get_dim() {return {hdr.dim[1], hdr.dim[2], hdr.dim[3]};};
	const char* get_filePath() const {return filePath.c_str();}; 
	float* get_pdataMatrix() {return dataMatrix;};

	float get_min() const {return minVal;};
	float get_max() const {return maxVal;};

};

#endif