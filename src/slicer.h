#ifndef SLICER_H
#define SLICER_H

#include "vector3.h"

class slicer
{

private:
	const float*  dataMatrix;
	vector3<int> sizeArray;
	vector3<int> slicePoint;

	float* planes[3]; // x normal, y normal, z normal
	bool isPlanesAlloc = 0;
	bool reqUpdate[3] = {0, 0, 0};
	bool flipFlag[3] = {0, 0, 1};

	void free_planes();
	void alloc_planes();
	void update_plane(const uint8_t iDim);

	
public:
	~slicer();

	void flip(const uint8_t iDim);
	float* get_plane(const uint8_t iDim);

	void set_sizeArray(const vector3<int> _sizeArray);
	void set_slicePoint(const vector3<int> _slicePoint);
	void set_dataMatrix(const float* _dataMatrix);
	vector3<int> get_slicePoint() const {return slicePoint;};
	vector3<int> get_sizeArray() const {return sizeArray;};

	bool* get_pflipFlag(const uint8_t iDim) {return &flipFlag[iDim];};
	bool get_flipFlag(const uint8_t iDim) const {return flipFlag[iDim];};

	void set_reqUpdate(const uint8_t iDim, const bool status) {reqUpdate[iDim] = status;};
};

#endif