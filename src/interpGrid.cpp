#include "interpGrid.h"

// returns a six element vector containing the indices of our positions
void interpGrid::getNeighbours(
	uint64_t* position, // input as requested position 
	uint64_t* neighbours, // indices of neighbouring voxels
	float* ratio){ // weighting of voxels
	
	float offsetDistance = 0; // distance from gridOrigin
	float leftDistance = 0; // distance from left neighbour
	float neighbourSpacing = 0; // spacing between two neighbours

	for(unsigned char iDim = 0; iDim < 3; iDim++){
		if (position[iDim] <= gridOrigin[iDim]){
			// case we are at the left hand end, set ratio to 0 and same pos
			ratio[iDim] = 0;
			neighbours[iDim * 2] = 0;
			neighbours[iDim * 2 + 1] = 0;
		}else if(position[iDim] >= gridEnd[iDim]){
			// case we are at the right hand end, set ratio to 0 and same pos
			ratio[iDim] = 0;
			neighbours[iDim * 2] = gridSize[iDim] - 1;
		 	neighbours[iDim * 2 + 1] =  gridSize[iDim] - 1;
		}else{
			// case we are anywhere in between
			offsetDistance = (float) position[iDim] - gridOrigin[iDim];
			neighbours[iDim * 2] = (uint64_t) offsetDistance / gridSpacing[iDim];
			neighbours[iDim * 2 + 1] = neighbours[iDim * 2] + 1;
			leftDistance = offsetDistance - (float) neighbours[iDim * 2] * gridSpacing[iDim];
			if(neighbours[iDim * 2 + 1] == (gridSize[iDim] - 1))
				neighbourSpacing = (gridSpacing[iDim] + remainder[iDim]) / 2;
			else
				neighbourSpacing = gridSpacing[iDim];

			ratio[iDim] = leftDistance / neighbourSpacing;
		}

		// for debugging and error checking
		if (ratio[iDim] > 1)
			printf("ratio above 1 should not exist\n");

		if (ratio[iDim] < 0)
			printf("ratios below 0 should not exist\n");
	}
	return;
}

// calculates the number of bins in each direction
void interpGrid::calcSubVols(){
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
void interpGrid::setGridSpacing(uint64_t* _gridSpacing){
	for(unsigned char i = 0; i < 3; i++)
		gridSpacing[i] = _gridSpacing[i];

	printf("[interpGrid] grid spacing: %ld, %ld, %ld\n", 
		gridSpacing[0], gridSpacing[1], gridSpacing[2]);

	return;
}

// defines the origin of the grid
void interpGrid::setGridOrigin(float* _gridOrigin){
	for (unsigned char i=0; i<3; i++)
		gridOrigin[i] = _gridOrigin[i];

	printf("[interpGrid] grid origin: %f, %f, %f\n", 
		gridOrigin[0], gridOrigin[1], gridOrigin[2]);
	return;
}

// defines the full size of the grid (not the number of bins)
void interpGrid::setVolumeSize(uint64_t* _volumeSize){
	for (unsigned char i = 0; i < 3; i++)
		volumeSize[i] = _volumeSize[i];

	printf("[interpGrid] gridSize: %ld, %ld, %ld\n", 
			volumeSize[0], volumeSize[1], volumeSize[2]);	
	return;
}
