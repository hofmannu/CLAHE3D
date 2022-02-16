#include "slicer.h"

slicer::~slicer()
{
	free_planes();
}

void slicer::free_planes()
{
	if (isPlanesAlloc)
	{
		for (uint8_t iDim = 0; iDim < 3; iDim++)
			delete[] planes[iDim];
	}
}

void slicer::alloc_planes()
{
	free_planes();

	planes[0] = new float[sizeArray.y * sizeArray.z];
	planes[1] = new float[sizeArray.x * sizeArray.z];
	planes[2] = new float[sizeArray.x * sizeArray.y];
	isPlanesAlloc = 1;
	return;
}

float* slicer::get_plane(const uint8_t iDim)
{
	update_plane(iDim);
	return planes[iDim];
}

void slicer::update_plane(const uint8_t iDim)
{
	if (reqUpdate[iDim] == 1)
	{
		if (iDim == 0) // lets update along yz plane
		{
			for (int iz = 0; iz < sizeArray.z; iz++)
			{
				const int zOffset = iz * sizeArray.x * sizeArray.y;
				for (int iy = 0; iy < sizeArray.y; iy++)
				{
					const int yOffset = iy * sizeArray.x;
					planes[iDim][iy + iz * sizeArray.y] = 
						dataMatrix[slicePoint.x + yOffset + zOffset];			
				}
			}
			reqUpdate[iDim] = 0;
		}
		else if (iDim == 1) // lets update along xz plane
		{
			const int yOffset = slicePoint.y * sizeArray.x;
			for (int iz = 0; iz < sizeArray.z; iz++)
			{
				const int zOffset = iz * sizeArray.x * sizeArray.y;
				for (int ix = 0; ix < sizeArray.x; ix++)
				{
					planes[iDim][ix + iz * sizeArray.x] = 
						dataMatrix[ix + yOffset + zOffset];			
				}
			}
			reqUpdate[iDim] = 0;
		}
		else if (iDim == 2) // let update along xy plane
		{
			const int zOffset = slicePoint.z * sizeArray.x * sizeArray.y;
			for (int iy = 0; iy < sizeArray.y; iy++)
			{
				const int yOffset = iy * sizeArray.x;
				for (int ix = 0; ix < sizeArray.x; ix++)
				{
					planes[iDim][ix + iy * sizeArray.x] = 
						dataMatrix[ix + yOffset + zOffset];			
				}
			}
			reqUpdate[iDim] = 0;
		} 

	}
}

void slicer::set_sizeArray(const vector3<int> _sizeArray)
{
	sizeArray = _sizeArray;
	reqUpdate[0] = 1; reqUpdate[1] = 1; reqUpdate[2] = 1;
	alloc_planes();
	return;
}

void slicer::set_slicePoint(const vector3<int> _slicePoint)
{
	#pragma unroll
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		if (slicePoint[iDim] != _slicePoint[iDim])
			reqUpdate[iDim] = 1;
	}
	slicePoint = _slicePoint;
	return;
}

void slicer::set_dataMatrix(const float* _dataMatrix)
{
	dataMatrix = _dataMatrix;
	reqUpdate[0] = 1; reqUpdate[1] = 1; reqUpdate[2] = 1;
	return;
}