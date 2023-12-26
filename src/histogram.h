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

public:
	// basic constructors and desctructors
	histogram();
	histogram(const std::size_t _nBins);
	~histogram();

	void calculate(const float* vector, const std::size_t nElems);

	// easy get functions
	[[nodiscard]] float get_minVal() const noexcept {return minVal;};
	[[nodiscard]] float get_maxVal() const noexcept {return maxVal;};
	[[nodiscard]] std::size_t get_minHist() const noexcept {return minHist;};
	[[nodiscard]] std::size_t get_maxHist() const noexcept {return maxHist;};
	[[nodiscard]] std::size_t get_idxPeak() const noexcept {return idxPeak;};
	[[nodiscard]] const float* get_pcontainerVal() const {return &containerVal[0];};
	[[nodiscard]] const std::size_t* get_pcounter() const {return &counter[0];};
	[[nodiscard]] std::vector<float> get_counter() const {std::vector<float> floatVec(counter.begin(), counter.end());return std::move(floatVec);};
	[[nodiscard]] int get_nBins() const {return nBins;};

	/// \brief prints the histogram result to a simple text file
	/// \param fileName name of file to be written
	void print_to_file(const std::string& fileName);
private:
	std::size_t nBins = 256;
	float minVal = 0.0f; //!< minimum of dataset
	float maxVal = 1.0f; //!< maximum of dataset
	std::size_t maxHist = 0; // maximum occurance in dataset (i.e. peak value)
	std::size_t minHist = 0; // minimum occurance in dataset (i.e. peak value)
	std::size_t idxPeak = 0;// index of m
	std::vector<float> containerVal; // center value of containers (x axis)
	std::vector<std::size_t> counter; // number of elements found (y axis)

	void alloc_mem();
	void free_mem();

	inline float get_binSize() const {return ((maxVal - minVal) / ((float) nBins));}; 
};

#endif