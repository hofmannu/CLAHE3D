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
	// work on local bounds so a reversed min/max does not permanently mutate the
	// member state (which would silently change the behaviour of the next call).
	float lo = minVal;
	float hi = maxVal;
	if (hi < lo) std::swap(lo, hi);
	for (uint64_t iElement = 0; iElement < nElements; iElement++)
	{
		if (array[iElement] < lo) array[iElement] = lo;
		if (array[iElement] > hi) array[iElement] = hi;
	}
}