#include <cstdint>

#ifndef THRESHOLDERSETT_H
#define THRESHOLDERSETT_H

struct thresholdersett
{
	float minVal = 0.0f;
	float maxVal = 1.0f;
};

#endif


#ifndef THRESHOLDER_H
#define THRESHOLDER_H

class thresholder
{
public:

	void threshold(float* array, uint64_t nElements);
	void set_minVal(float _minVal) noexcept;
	void set_maxVal(float _maxVal) noexcept;
private:
	float minVal = 0.0f;
	float maxVal = 1.0f;


};

#endif