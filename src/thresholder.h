#include <cstdint>

#ifndef THRESHOLDERSETT_H
#define THRESHOLDERSETT_H

struct thresholdersett
{
	float minVal = 0;
	float maxVal = 1;
};

#endif


#ifndef THRESHOLDER_H
#define THRESHOLDER_H

class thresholder
{
private:
	float minVal = 0;
	float maxVal = 1;
public:

	void threshold(float* array, const uint64_t nElements);
	void set_minVal(const float _minVal);
	void set_maxVal(const float _maxVal);

};

#endif