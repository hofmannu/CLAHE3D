#include "gridder.h"

// defines the size of the individual subvolumes (lets make this uneven)
void gridder::set_sizeSubVols(const vector3<int64_t>& _subVolSize)
{
	
	#pragma unroll
	for(uint8_t iDim = 0; iDim < 3; iDim++)
	{
		if ((_subVolSize[iDim] % 2) == 0)
		{
			printf("Please choose the size of the subvolumes uneven");
			throw "InvalidValue";
		}
	}
	sizeSubVols = _subVolSize;
	origin = (sizeSubVols - 1) / 2;
	return;
}

// defines the spacing of the individual histogram samples
void gridder::set_spacingSubVols(const vector3<int64_t>& _spacingSubVols)
{
	if (_spacingSubVols.any(0))
	{
			printf("The spacing of the subvolumes needs to be at least 1");
			throw "InvalidValue";
	}
	spacingSubVols = _spacingSubVols;
	return;
}

// size of full three dimensional volume
void gridder::set_volSize(const vector3<int64_t>& _volSize)
{
	if (_volSize.any(0))
	{
		printf("The size of the volume should be bigger then 0");
		throw "InvalidValue";
	}
	volSize = _volSize;
	
	nElements = volSize.x * volSize.y * volSize.z;

	return;
}

// calculate number of subvolumes
void gridder::calculate_nsubvols()
{
	// number of subvolumes
	// # pragma unroll
	// for (unsigned char iDim = 0; iDim < 3; iDim++)
	// {
	// 	const int64_t lastIdx = volSize[iDim] - 1;
	// 	nSubVols[iDim] = (lastIdx - origin[iDim]) / spacingSubVols[iDim] + 1;
	// 	endIdx[iDim] = origin[iDim] + (nSubVols[iDim] - 1) * spacingSubVols[iDim];
	// }
	vector3<int64_t> lastIdx = volSize - 1;
	nSubVols = (lastIdx - origin) / spacingSubVols + 1;

	return;
}

//
void gridder::get_neighbours(
	const vector3<int64_t>& position,
	int64_t* neighbours,
	float* ratio) const
{
	#pragma unroll
	for (uint8_t iDim = 0; iDim < 3; iDim++)
	{
		// let see if we hit the lower limit
		if (((float) position[iDim]) <=  origin[iDim])
		{
			ratio[iDim] = 0;
			neighbours[iDim * 2] = 0; // left index along current dimension
			neighbours[iDim * 2 + 1] = 0; // right index along current dimension
		}
		else if (((float) position[iDim]) >=  endIdx[iDim])
		{
			ratio[iDim] = 0;
			neighbours[iDim * 2] =  nSubVols[iDim] - 1; // left index along curr dimension
		 	neighbours[iDim * 2 + 1] = nSubVols[iDim] - 1; // right index along curr dimension
		} 
		else // we are actually in between!
		{
			const float offsetDistance = ((float) position[iDim]) - (float) origin[iDim];
			neighbours[iDim * 2] = (int64_t) (offsetDistance / spacingSubVols[iDim]);
			neighbours[iDim * 2 + 1] = neighbours[iDim * 2] + 1;
			const float leftDistance = offsetDistance - ((float) neighbours[iDim * 2]) * 
				((float) spacingSubVols[iDim]);
			ratio[iDim] = leftDistance / ((float) spacingSubVols[iDim]);
		}
	}
	return;
}

vector3<int64_t> gridder::get_startIdxSubVol(const vector3<int64_t> iSub) const 
{
	const vector3<int64_t> centerPos = iSub * spacingSubVols + origin;
	vector3<int64_t> startIdx = centerPos - (sizeSubVols - 1) / 2; 
	
	if (startIdx.x < 0)
		startIdx.x = 0;

	if (startIdx.y < 0)
			startIdx.y = 0;

	if (startIdx.z < 0)
			startIdx.z = 0;

	return startIdx;
}

vector3<int64_t> gridder::get_stopIdxSubVol(const vector3<int64_t> iSub) const
{
	const vector3<int64_t> centerPos = iSub * spacingSubVols + origin;
	vector3<int64_t> stopIdx = centerPos + (sizeSubVols - 1) / 2; 

	
	if (stopIdx.x >= volSize.x)
		stopIdx.x = volSize.x - 1;

	if (stopIdx.y >= volSize.y)
			stopIdx.y = volSize.y - 1;

	if (stopIdx.z >= volSize.z)
			stopIdx.z = volSize.z - 1;

	return stopIdx;
}

// returns number of element in the volume
int64_t gridder::get_nElements() const
{
	return nElements;
}