#include "gridder.h"

// defines the size of the individual subvolumes (lets make this uneven)
void gridder::set_sizeSubVols(const uint64_t* _subVolSize)
{
	#pragma unroll
	for(uint8_t iDim = 0; iDim < 3; iDim++)
	{
		if ((_subVolSize[iDim] % 2) == 0)
		{
			printf("Please choose the size of the subvolumes uneven");
			throw "InvalidValue";
		}
		sizeSubVols[iDim] = _subVolSize[iDim];
		origin[iDim] = (sizeSubVols[iDim] - 1) / 2;
	}
	return;
}

// defines the spacing of the individual histogram samples
void gridder::set_spacingSubVols(const uint64_t* _spacingSubVols)
{
	#pragma unroll
	for(uint8_t iDim = 0; iDim < 3; iDim++)
	{
		if (_spacingSubVols[iDim] == 0)
		{
			printf("The spacing of the subvolumes needs to be at least 1");
			throw "InvalidValue";
		}
		spacingSubVols[iDim] = _spacingSubVols[iDim];
	}


	return;
}

// size of full three dimensional volume
void gridder::set_volSize(const uint64_t* _volSize)
{
	#pragma unroll
	for(uint8_t iDim = 0; iDim < 3; iDim++)
	{
		if (_volSize[iDim] == 0)
		{
			printf("The size of the volume should be bigger then 0");
			throw "InvalidValue";
		}
		volSize[iDim] = _volSize[iDim];
	}
	nElements = volSize[0] * volSize[1] * volSize[2];

	return;
}

// calculate number of subvolumes
void gridder::calculate_nsubvols()
{
	// number of subvolumes
	# pragma unroll
	for (unsigned char iDim = 0; iDim < 3; iDim++)
	{
		const uint64_t lastIdx = volSize[iDim] - 1;
		nSubVols[iDim] = (lastIdx - origin[iDim]) / spacingSubVols[iDim] + 1;
		endIdx[iDim] = origin[iDim] + (nSubVols[iDim] - 1) * spacingSubVols[iDim];
	}

	return;
}

//
void gridder::get_neighbours(
	const uint64_t* position,
	uint64_t* neighbours,
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
			neighbours[iDim * 2] = (uint64_t) (offsetDistance / spacingSubVols[iDim]);
			neighbours[iDim * 2 + 1] = neighbours[iDim * 2] + 1;
			const float leftDistance = offsetDistance - ((float) neighbours[iDim * 2]) * 
				((float) spacingSubVols[iDim]);
			ratio[iDim] = leftDistance / ((float) spacingSubVols[iDim]);
		}
	}
	return;
}

uint64_t gridder::get_startIdxSubVol(const uint64_t iSub, const uint8_t iDim) const 
{
	const int64_t centerPos = (int64_t) iSub * spacingSubVols[iDim] + origin[iDim];
	int64_t startIdx = centerPos - ((int) sizeSubVols[iDim] - 1) / 2; 
	startIdx = (startIdx < 0) ? 0 : startIdx;
	return (uint64_t) startIdx;
}

uint64_t gridder::get_stopIdxSubVol(const uint64_t iSub, const uint8_t iDim) const
{
	const int64_t centerPos = (int64_t) iSub * spacingSubVols[iDim] + origin[iDim];
	int64_t stopIdx = centerPos + ((int) sizeSubVols[iDim] - 1) / 2; 
	stopIdx = (stopIdx >= volSize[iDim]) ? (volSize[iDim] - 1) : stopIdx;
	return (uint64_t) stopIdx;
}

// returns number of element in the volume
uint64_t gridder::get_nElements() const
{
	return nElements;
}