/*
	a class describing the histogram of a dataset
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 08.03.2022

	Changelog:
		- 2022-04-06: moved to std::size_t
*/

#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <cstdlib>
#include <string>
#include <fstream>
#include <vector>

class histogram
{
private:
	std::size_t nBins = 256;
	
	float minVal = 0; // minimum of dataset
	float maxVal = 1; // maximum of dataset
	// note: represent the x axis of the histogram

	std::size_t maxHist = 0; // maximum occurance in dataset (i.e. peak value)
	std::size_t minHist = 0; // minimum occurance in dataset (i.e. peak value)
	std::size_t idxPeak = 0;// index of m

	std::vector<float> containerVal; // center value of containers (x axis)
	std::vector<std::size_t> counter; // number of elements found (y axis)

	void alloc_mem();
	void free_mem();

	inline float get_binSize() const {return ((maxVal - minVal) / ((float) nBins));}; 
public:


	// basic constructors and desctructors
	histogram();
	histogram(const std::size_t _nBins);
	~histogram();

	void calculate(const float* vector, const std::size_t nElems);

	// easy get functions
	float get_minVal() const {return minVal;};
	float get_maxVal() const {return maxVal;};
	std::size_t get_minHist() const {return minHist;};
	std::size_t get_maxHist() const {return maxHist;};
	std::size_t get_idxPeak() const {return idxPeak;};
	const float* get_pcontainerVal() const {return &containerVal[0];};
	const std::size_t* get_pcounter() const {return &counter[0];};
	std::vector<float> get_counter() const {std::vector<float> floatVec(counter.begin(), counter.end());return std::move(floatVec);};
	int get_nBins() const {return nBins;};

	void print_to_file(const std::string fileName);

};

#endif