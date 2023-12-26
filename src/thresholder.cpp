#include "thresholder.h"
#include <algorithm>

void thresholder::set_minVal(const float _minVal) noexcept
{
	minVal = _minVal;
}

void thresholder::set_maxVal(const float _maxVal) noexcept
{
	maxVal = _maxVal;
}

// apply actual threshold to volume
void thresholder::threshold(float* array, const uint64_t nElements)
{
	if (maxVal < minVal) std::swap(minVal, maxVal);
	for (uint64_t iElement = 0; iElement < nElements; iElement++)
	{
		if (array[iElement] < minVal) array[iElement] = minVal;
		if (array[iElement] > maxVal) array[iElement] = maxVal;
	}
}