/* 
	small helper class used to check errors, analyzed available memory etc
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 13.02.2022
*/

#ifndef CUDATOOLS_H
#define CUDATOOLS_H
	
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

class cudaTools
{
	private:
		int nDevices = 0;
		vector<cudaDeviceProp> props;
	public:
		cudaTools();

		void checkCudaErr(const cudaError_t& err, const char* msgErr);
		cudaError_t cErr;

		int get_nDevices() const {return nDevices;};

		void print_devProps(const int iDevice);
		void print_devProps();
		cudaDeviceProp get_devProps(const int iDevice) {return props[iDevice];};

};

#endif