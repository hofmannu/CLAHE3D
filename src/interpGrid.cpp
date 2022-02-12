#include "interpGrid.h"

// returns a six element vector containing the indices of our positions
// neighbours: [zLower, zUpper, xLower, xUpper, yLower, yUpper]
void interpGrid::getNeighbours(
	const uint64_t* position, // input as requested position 
	uint64_t* neighbours, // indices of neighbouring voxels
	float* ratio) // weighting of voxels
{ 
	
	float offsetDistance = 0; // distance from gridOrigin
	float leftDistance = 0; // distance from left neighbour
	float neighbourSpacing = 0; // spacing between two neighbours

	#pragma unroll
	for(uint8_t iDim = 0; iDim < 3; iDim++)
	{
		if (((float) position[iDim]) <= gridOrigin[iDim])
		{
			// printf("Hitting lower end");
			// case we are at the left hand end, set ratio to 0 and same pos
			ratio[iDim] = 0;
			neighbours[iDim * 2] = 0;
			neighbours[iDim * 2 + 1] = 0;
		}
		else if(((float) position[iDim]) >= gridEnd[iDim])
		{
			// printf("Hitting upper end");
			// case we are at the right hand end, set ratio to 0 and same pos
			ratio[iDim] = 0;
			neighbours[iDim * 2] = gridSize[iDim] - 1;
		 	neighbours[iDim * 2 + 1] =  gridSize[iDim] - 1;
		}
		else
		{
			// printf("in between");
			// case we are anywhere in between
			offsetDistance = (float) position[iDim] - gridOrigin[iDim];
			neighbours[iDim * 2] = (uint64_t) offsetDistance / gridSpacing[iDim];
			neighbours[iDim * 2 + 1] = neighbours[iDim * 2] + 1;
			leftDistance = offsetDistance - (float) neighbours[iDim * 2] * gridSpacing[iDim];
			
			if (neighbours[iDim * 2 + 1] == (gridSize[iDim] - 1))
			{
				neighbourSpacing = (gridSpacing[iDim] + remainder[iDim]) / 2;
			}
			else
			{
				neighbourSpacing = gridSpacing[iDim];
			}

			ratio[iDim] = leftDistance / neighbourSpacing;
		}

	}
	return;
}

// calculates the number of bins in each direction
void interpGrid::calcSubVols()
{
	uint64_t volumeLength;
	for (unsigned char iDim = 0; iDim < 3; iDim++){
		volumeLength = volumeSize[iDim] - 1; // length of full volume in units
		// calculate grid size (number of elements}
		gridSize[iDim] = (volumeLength - 1) / gridSpacing[iDim] + 1;
		// calculate size of last element (remainder)
		remainder[iDim] = (float) volumeLength - ((float) gridSize[iDim] - 1.0) * gridSpacing[iDim]; 
		gridEnd[iDim] = (float) gridSpacing[iDim] * ((float) gridSize[iDim] - 1.0) + remainder[iDim] / 2; 
	}	

	printf("[interpGrid] number of subvolumes: %d, %d, %d",
		gridSize[0], gridSize[1], gridSize[2]);
	return;
}

// sets the required spacing of grid
void interpGrid::setGridSpacing(const uint64_t* _gridSpacing)
{
	for(uint8_t iDim = 0; iDim < 3; iDim++)
	{
		gridSpacing[iDim] = _gridSpacing[iDim];
	}

	printf("[interpGrid] grid spacing: %ld, %ld, %ld\n", 
		gridSpacing[0], gridSpacing[1], gridSpacing[2]);

	return;
}

// defines the origin of the grid
void interpGrid::setGridOrigin(const float* _gridOrigin)
{
	for (unsigned char i=0; i<3; i++)
		gridOrigin[i] = _gridOrigin[i];

	printf("[interpGrid] grid origin: %f, %f, %f\n", 
		gridOrigin[0], gridOrigin[1], gridOrigin[2]);
	return;
}

// defines the full size of the grid (not the number of bins)
void interpGrid::setVolumeSize(const uint64_t* _volumeSize){
	for (unsigned char i = 0; i < 3; i++)
		volumeSize[i] = _volumeSize[i];

	printf("[interpGrid] volumeSize: %ld, %ld, %ld\n", 
			volumeSize[0], volumeSize[1], volumeSize[2]);	
	return;
}