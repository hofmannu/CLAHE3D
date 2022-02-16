#include "histogram.h"

histogram::histogram()
{

}

histogram::histogram(const int _nBins)
{
	nBins = _nBins;
	alloc_mem();
}

histogram::~histogram()
{
	free_mem();
}

void histogram::alloc_mem()
{
	free_mem();

	containerVal = new float[nBins];
	counter = new float[nBins];
	isMemAlloc = 1;
	return;
}

void histogram::free_mem()
{
	if (isMemAlloc)
	{
		delete[] containerVal;
		delete[] counter;
	}
	return;
}

void histogram::calculate(const float* vector, const int nElems)
{
	if (!isMemAlloc)
		alloc_mem();

	// get min and max first
	minVal = vector[0];
	maxVal = vector[1];
	for (int iElem = 0; iElem < nElems; iElem++)
	{
		if (vector[iElem] < minVal)
		{
			minVal = vector[iElem];
		}
		if (vector[iElem] > maxVal)
		{
			maxVal = vector[iElem];
		}
	
	}

	// set bins all to zero
	const float binSize = get_binSize();
	for (int iBin = 0; iBin < nBins; iBin ++)
	{
		counter[iBin] = 0;
		containerVal[iBin] = (0.5 + ((float ) iBin)) * binSize; 
	}
	
	for (int iElem = 0; iElem < nElems; iElem++) 
	{
		int currBin = int((vector[iElem] - minVal) / binSize);
		if (currBin >= nBins)
			currBin = nBins - 1;

		counter[currBin]++;
	}

	// lets leave the first element out for max of histogram
	minHist = counter[0];
	maxHist = counter[1];
	for (int iHist = 1; iHist < nBins; iHist++)
	{
		if (counter[iHist] > maxHist)
			maxHist = counter[iHist];

		// lets ommit 
		if (counter[iHist] < minHist)
			minHist = counter[iHist];
	}

}

