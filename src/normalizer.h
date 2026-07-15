/*
	File: normalizer.h
	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 10.03.2022
*/


// struct holding settings for our normalizer
#ifndef NORMALIZERSETT_H
#define NORMALIZERSETT_H

template <class F>
struct normalizersett
{
	F minVal = 0.0f;
	F maxVal = 1.0f;
};

#endif


// class performing normalization
#ifndef NORMALIZER_H
#define NORMALIZER_H

#include <cstdint>

template <class F>
class normalizer
{

private:
	F minVal = 0;
	F maxVal = 1;
public:

	// get functions
	F get_minVal() const {return minVal;};
	F get_maxVal() const {return maxVal;};

	// set functions
	void set_minVal(const F _minVal)
	{
		minVal = _minVal;
		return;
	}

	void set_maxVal(const F _maxVal)
	{
		maxVal = _maxVal;
		return;
	}

	// recalculating data span
	template<typename T>
	void normalize(F* array, const T nElements)
	{
		if (nElements > 0)
		{
			const F range = maxVal - minVal;

			// first we need to find current maximum and minimum in array
			// (both seeded from element 0 - seeding max from array[1] would read
			// out of bounds for a single-element array)
			F minValArray = array[0];
			F maxValArray = array[0];
			for (T iElement = 0; iElement < nElements; iElement++)
			{
				if (array[iElement] > maxValArray)
					maxValArray = array[iElement];

				if (array[iElement] < minValArray)
					minValArray = array[iElement];
			}
			const F rangeArray = maxValArray - minValArray;

			// a constant array has zero span: 1 / rangeArray would be inf and
			// (value - min) * inf would be NaN. Map every element to minVal instead.
			if (rangeArray == 0)
			{
				for (T iElement = 0; iElement < nElements; iElement++)
					array[iElement] = minVal;
				return;
			}

			const F irangeArray = 1.0f / rangeArray;

			for (T iElement = 0; iElement < nElements; iElement++)
			{
				const F value01 = (array[iElement] - minValArray) * irangeArray;
				// now it spans 0 to 1 range
				array[iElement] = value01 * range + minVal;
				// now it should span new range
			}

		}

		return;
	}
};

#endif