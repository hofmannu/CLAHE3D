#include "slicer.h"


// constructor and destructor
slicer::slicer()
{

}

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
	if (!isPlanesAlloc)
		alloc_planes();

	update_plane(iDim);
	return planes[iDim];
}

void slicer::update_plane(const uint8_t iDim)
{
	if (reqUpdate[iDim] == 1)
	{
		if (iDim == 0) // lets update along yz plane [iy, iz]
		{
			for (std::size_t iz = 0; iz < sizeArray.z; iz++)
			{
				const std::size_t zOut = (flipFlag[2]) ? (sizeArray.z - 1 - iz) : iz;
				const std::size_t zOffset = iz * sizeArray.x * sizeArray.y;
				for (std::size_t iy = 0; iy < sizeArray.y; iy++)
				{
					const std::size_t yOut = (flipFlag[1]) ? (sizeArray.y - 1 - iy) : iy;
					const std::size_t yOffset = iy * sizeArray.x;
					planes[iDim][yOut + zOut * sizeArray.y] = 
						dataMatrix[slicePoint.x + yOffset + zOffset];
				}
			}
		}
		else if (iDim == 1) // lets update along xz plane [ix, iz]
		{
			const std::size_t yOffset = slicePoint.y * sizeArray.x;
			for (std::size_t iz = 0; iz < sizeArray.z; iz++)
			{
				const std::size_t zOut = (flipFlag[2]) ? (sizeArray.z - 1 - iz) : iz;
				const std::size_t zOffset = iz * sizeArray.x * sizeArray.y;
				for (std::size_t ix = 0; ix < sizeArray.x; ix++)
				{
					const std::size_t xOut = (flipFlag[0]) ? (sizeArray.x - 1 - ix) : ix;
					planes[iDim][zOut * sizeArray.x + xOut] = 
						dataMatrix[ix + yOffset + zOffset];			
				}
			}
		}
		else if (iDim == 2) // let update along xy plane [ix, iy]
		{
			const std::size_t zOffset = slicePoint.z * sizeArray.x * sizeArray.y;
			for (std::size_t iy = 0; iy < sizeArray.y; iy++)
			{
				const std::size_t yOut = (!flipFlag[1]) ? (sizeArray.y - 1 - iy) : iy;
				const std::size_t yOffset = iy * sizeArray.x;
				for (std::size_t ix = 0; ix < sizeArray.x; ix++)
				{
					const std::size_t xOut = (flipFlag[0]) ? (sizeArray.x - 1 - ix) : ix;
					planes[iDim][xOut + yOut * sizeArray.x] = 
						dataMatrix[ix + yOffset + zOffset];			
				}
			}
		} 
		reqUpdate[iDim] = 0;

	}
}

void slicer::set_sizeArray(const vector3<std::size_t> _sizeArray)
{
	sizeArray = _sizeArray;
	reqUpdate[0] = 1; reqUpdate[1] = 1; reqUpdate[2] = 1;
	alloc_planes();
	return;
}

// define slice point through a vector
void slicer::set_slicePoint(const vector3<std::size_t> _slicePoint)
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

// define through three individual values
void slicer::set_slicePoint(const std::size_t ix, const std::size_t iy, const std::size_t iz)
{
	if (slicePoint[0] != ix)
		reqUpdate[0] = 1;
	slicePoint.x = ix;

	if (slicePoint[1] != iy)
		reqUpdate[1] = 1;
	slicePoint.y = iy;

	if (slicePoint[2] != iz)
		reqUpdate[2] = 1;
	slicePoint.z = iz;
	return;
}

void slicer::set_dataMatrix(const float* _dataMatrix)
{
	dataMatrix = _dataMatrix;
	reqUpdate[0] = 1; reqUpdate[1] = 1; reqUpdate[2] = 1;
	return;
}

void slicer::flip(const uint8_t iDim)
{
	flipFlag[iDim] = !flipFlag[iDim];
	reqUpdate[iDim] = 1;
	update_plane(iDim);
	return;
}