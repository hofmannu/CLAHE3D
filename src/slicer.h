/*
	a class used to generate slices from a volume

	Author: Urs Hofmann
	Mail: mail@hofmannu.org
	Date: 06.04.2022

*/

#ifndef SLICER_H
#define SLICER_H

#include "vector3.h"

class slicer
{

private:
	const float*  dataMatrix;
	vector3<std::size_t> sizeArray;
	vector3<std::size_t> slicePoint;

	float* planes[3]; // x normal, y normal, z normal
	bool isPlanesAlloc = 0;
	bool reqUpdate[3] = {0, 0, 0};
	bool flipFlag[3] = {0, 0, 1};

	void free_planes();
	void alloc_planes();
	void update_plane(const uint8_t iDim);

	
public:
	slicer();
	~slicer();

	void flip(const uint8_t iDim);
	float* get_plane(const uint8_t iDim);

	void set_sizeArray(const vector3<std::size_t> _sizeArray);

	void set_slicePoint(const vector3<std::size_t> _slicePoint);
	void set_slicePoint(const std::size_t ix, const std::size_t iy, const std::size_t iz);
	
	void set_dataMatrix(const float* _dataMatrix);
	vector3<std::size_t> get_slicePoint() const {return slicePoint;};
	vector3<std::size_t> get_sizeArray() const {return sizeArray;};

	bool* get_pflipFlag(const uint8_t iDim) {return &flipFlag[iDim];};
	bool get_flipFlag(const uint8_t iDim) const {return flipFlag[iDim];};

	void set_reqUpdate(const uint8_t iDim, const bool status) {reqUpdate[iDim] = status;};
};

#endif