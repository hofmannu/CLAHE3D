#include "thresholder.h"

void thresholder::set_minVal(const float _minVal)
{
	minVal = _minVal;
	return;
}

void thresholder::set_maxVal(const float _maxVal)
{
	maxVal = _maxVal;
	return;
}

// apply actual threshold to volume
void thresholder::threshold(float* array, const uint64_t nElements)
{
	#pragma unroll
	for (uint64_t iElement = 0; iElement < nElements; iElement++)
	{
		if (array[iElement] < minVal)
			array[iElement] = minVal;

		if (array[iElement] > maxVal)
			array[iElement] = maxVal;

	}
	return;
}