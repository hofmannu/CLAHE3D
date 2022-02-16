#ifndef HISTOGRAM_H
#define HISTOGRAM_H

class histogram
{
private:
	int nBins = 256;
	float minVal = 0; // minimum of dataset
	float maxVal = 1; // maximum of dataset
	float maxHist = 0;
	float minHist = 0;

	float* containerVal; // center value of containers
	float* counter; // number of elements found
	bool isMemAlloc = 0;

	void alloc_mem();
	void free_mem();

	inline float get_binSize() const {return ((maxVal - minVal) / ((float) nBins));}; 
public:

	histogram();
	histogram(const int _nBins);
	~histogram();


	void calculate(const float* vector, const int nElems);

	// easy get functions
	float get_minVal() const {return minVal;};
	float get_maxVal() const {return maxVal;};
	float get_minHist() const {return minHist;};
	float get_maxHist() const {return maxHist;};
	float* get_pcontainerVal() {return containerVal;};
	float* get_pcounter() {return counter;};
	int get_nBins() const {return nBins;};

};

#endif