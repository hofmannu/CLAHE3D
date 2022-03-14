/*
	a class to run a medianfilter over a threedimensional volume
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 14.03.2022
*/

#include "genfilt.h"
#include "vector3.h"
#include <cstring>
#include <vector>
#include <algorithm>
#include <thread>

#ifndef MEDIANFILTSETT_H
#define MEDIANFILTSETT_H

struct medianfiltsett
{
	int kernelSize[3] = {3, 3, 3};
};

#endif


#ifndef MEDIANFILT_H
#define MEDIANFILT_H

class medianfilt : public genfilt
{
private:
	// int nThreads = 1;
	vector<int> zStart; // start line for each thread
	vector<int> zStop; // stop line for each thread
	int nKernel; // number of elements in kernel
	int centerIdx; // center index of linearized kernel
	int sizeKernel; // size of kernel in bytes
	
	void padd(); // function to apply padding to dataset
	void run_range(const int iRange); // run for a certain z range (multithread)
public:
	medianfilt(); // class constructor

	void run();
};

#endif