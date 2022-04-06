#include "histogram.h"


// constructors and descrructors
histogram::histogram()
{

}

histogram::histogram(const std::size_t _nBins)
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
	containerVal.resize(nBins);
	counter.resize(nBins);
	return;
}

void histogram::free_mem()
{
	containerVal.clear();
	counter.clear();
	return;
}

void histogram::calculate(const float* vector, const std::size_t nElems)
{
	alloc_mem();

	// get min and max first
	minVal = vector[0];
	maxVal = vector[0];
	for (std::size_t iElem = 0; iElem < nElems; iElem++)
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
	for (std::size_t iBin = 0; iBin < nBins; iBin ++)
	{
		counter[iBin] = 0; // reset all counters to 0
		containerVal[iBin] = minVal + (0.5 + ((float ) iBin)) * binSize; 
	}
	
	for (std::size_t iElem = 0; iElem < nElems; iElem++) 
	{
		std::size_t currBin = (vector[iElem] - minVal) / binSize;
		if (currBin >= nBins)
			currBin = nBins - 1;

		counter[currBin]++;
	}

	// lets leave the first element out for max of histogram
	minHist = counter[0];
	maxHist = counter[0];
	idxPeak = 0;
	for (std::size_t iHist = 1; iHist < nBins; iHist++)
	{
		if (counter[iHist] > maxHist)
		{
			maxHist = counter[iHist];
			idxPeak = iHist;
		}

		// lets ommit 
		if (counter[iHist] < minHist)
		{
			minHist = counter[iHist];
		}
	}

	return;
}

void histogram::print_to_file(const std::string fileName)
{

	const float binSize = get_binSize();
	std::ofstream myfile;
  myfile.open (fileName.c_str());

  myfile << "nBins = " << nBins << std::endl;
  myfile << "Bin size = " << binSize << std::endl;
  myfile << "Data range = " << minVal << " ... " << maxVal << std::endl;
  myfile << "Occurance range = " << minHist << " ... " << maxHist << std::endl;

  myfile << "\n\nIndex | Center value | Lower bound | Upper bound | Occurance \n";
  for (std::size_t idx = 0; idx < nBins; idx++)
  {
  	myfile << 
  		idx << " | " << 
  		containerVal[idx] << " | " << 
  		containerVal[idx] - binSize * 0.5f << " | " <<  
  		containerVal[idx] + binSize * 0.5f << " | " << 
  		counter[idx] << std::endl;
  }

  myfile.close();
	return;
}