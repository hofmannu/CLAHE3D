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

using namespace std;

class cudaTools
{
	public:
		void checkCudaErr(cudaError_t err, const char* msgErr);
};

#endif