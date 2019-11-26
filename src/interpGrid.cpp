#include "interpGrid.h"

// returns a six element vector containing the indices of our positions
void interpGrid::getNeighbours(
	uint64_t* position, // input as requested position 
	uint64_t* neighbours, // indices of neighbouring voxels
	float* ratio){ // weighting of voxels
	
	for(unsigned char iDim = 0; iDim < 3; iDim++){
		if (position[iDim] < gridOrigin[iDim]){
			// case we are at the left hand corner, set ratio to 0 and same pos
			ratio[iDim] = 0;
			neighbours[iDim * 2] = 0;
			neighbours[iDim * 2 + 1] = 0;
		}else if(position[iDim] > (gridOrigin[iDim] + gridSpacing[iDim] * (gridSize[iDim] - 1))){
			ratio[iDim] = 0;
			neighbours[iDim * 2] = gridSize[iDim] - 1;
		 	neighbours[iDim * 2 + 1] =  gridSize[iDim] - 1;
		}else{
			neighbours[iDim * 2] = ((float) position[iDim] - gridOrigin[iDim]) / gridSpacing[iDim];
			neighbours[iDim * 2 + 1] = neighbours[iDim * 2] + 1;
			ratio[iDim] = ((float) position[iDim] - gridOrigin[iDim]) - (float) neighbours[iDim * 2] * gridSpacing[iDim]; // should now range from 0 to 1
			ratio[iDim] = ratio[iDim] / gridSpacing[iDim];
		}

		// for debugging and error checking
		// if (ratio[iDim] > 1)
		//	printf("ratio above 1 should not exist\n");

		// if (ratio[iDim] < 0)
		//	printf("ratios below 0 should not exist\n");
	}
	return;
}

void interpGrid::setGridSpacing(uint64_t* _gridSpacing){
	for(unsigned char i = 0; i < 3; i++)
		gridSpacing[i] = _gridSpacing[i];

	printf("[interpGrid] grid spacing: %ld, %ld, %ld\n", 
		gridSpacing[0], gridSpacing[1], gridSpacing[2]);

	return;
}

void interpGrid::setGridOrigin(float* _gridOrigin){
	for (unsigned char i=0; i<3; i++)
		gridOrigin[i] = _gridOrigin[i];

	printf("[interpGrid] grid origin: %f, %f, %f\n", 
		gridOrigin[0], gridOrigin[1], gridOrigin[2]);
	return;
}


void interpGrid::setGridSize(uint64_t* _gridSize){
	for (unsigned char i = 0; i < 3; i++)
		gridSize[i] = _gridSize[i];

	printf("[interpGrid] gridSize: %ld, %ld, %ld\n", 
			gridSize[0], gridSize[1], gridSize[2]);	
	return;
}
